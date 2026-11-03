# ruff: noqa: F811 - pytest fixtures imported from test_api_fixtures
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_api_fixtures import FakeRunner, client, home, seq  # noqa: F401

from mtmct.config import paths
from mtmct.data.base import ADAPTERS, DatasetMeta
from mtmct.db import Dataset, Job, open_session
from mtmct.worker.main import Worker
from mtmct.worker.media import ensure_media


class _FakeAdapter:
    id = "fakeds"

    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            id="fakeds",
            title="Fake",
            licence="MIT",
            source_url="http://x",
            overlap="overlap",
            real=False,
            n_cameras="2",
            download_gb=0.1,
            sequences=["seq"],
        )

    def fetch(self, raw_dir: Path, progress) -> None:
        progress(0.5, "half")
        (raw_dir / "f.txt").parent.mkdir(parents=True, exist_ok=True)
        (raw_dir / "f.txt").write_text("x")

    def prepare(self, raw_dir: Path, data_root: Path, progress):
        from mtmct.data.base import load_sequence

        progress(1.0, "prepared")
        return [load_sequence(data_root, "toy", "seq")]


@pytest.fixture
def fake_adapter():
    ADAPTERS["fakeds"] = _FakeAdapter()
    yield
    ADAPTERS.pop("fakeds", None)


def test_media_range_request_206(client, seq):
    ensure_media(seq)
    full = client.get("/api/media/toy/seq/c0.mp4")
    assert (
        full.status_code == 200
        and full.headers["accept-ranges"] == "bytes"
        and full.headers["content-type"] == "video/mp4"
    )
    part = client.get("/api/media/toy/seq/c0.mp4", headers={"Range": "bytes=10-99"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 10-99/{len(full.content)}"
    assert part.content == full.content[10:100]
    assert client.get("/api/media/toy/seq/c0.jpg").headers["content-type"] == "image/jpeg"


def test_paths_not_taken_from_client(client, seq, home):
    secret = home / "secret.mp4"
    secret.write_bytes(b"top secret")
    for url in (
        "/api/media/toy/../../secret/c0.mp4",
        "/api/media/toy/seq/..%2F..%2F..%2Fsecret.mp4",
        "/api/media/toy/seq/zz.mp4",
        "/api/media/..%2Fx/seq/c0.mp4",
        "/api/media/toy/seq/c0.exe",
    ):
        r = client.get(url)
        assert r.status_code in (400, 404, 422), url
        assert b"top secret" not in r.content


def _mp4_bytes(tmp_path: Path) -> bytes:
    from conftest import write_video

    from mtmct.worker.media import make_proxy

    src = tmp_path / "up.avi"
    write_video(src, 10, 10.0)
    make_proxy(src, tmp_path / "up.mp4")
    return (tmp_path / "up.mp4").read_bytes()


def test_upload_custom_rejects_non_video(client, home):
    r = client.post(
        "/api/datasets/custom",
        data={"name": "bad"},
        files=[("videos", ("../../evil.mp4", io.BytesIO(b"not a video at all"), "video/mp4"))],
    )
    assert r.status_code == 415
    assert not (paths().raw / "custom" / "bad").exists()
    assert not list(home.rglob("evil*"))
    assert (
        client.post(
            "/api/datasets/custom", data={"name": "../x"}, files=[("videos", ("a.mp4", io.BytesIO(b"x"), "video/mp4"))]
        ).status_code
        == 422
    )


def test_upload_custom_accepts_video_with_safe_names(client, home, tmp_path):
    data = _mp4_bytes(tmp_path)
    r = client.post(
        "/api/datasets/custom",
        data={"name": "mine"},
        files=[
            ("videos", ("../../../evil name.mp4", io.BytesIO(data), "video/mp4")),
            ("videos", ("b.mov", io.BytesIO(data), "video/quicktime")),
            ("calibration", ("calib.json", io.BytesIO(b'{"c0": {}}'), "application/json")),
            ("gt", ("gt.csv", io.BytesIO(b"camera_id,frame_id,person_id,x,y,w,h\nc1,0,1,1,1,5,5\n"), "text/csv")),
        ],
    )
    assert r.status_code == 200, r.text
    raw = paths().raw / "custom" / "mine"
    assert sorted(p.name for p in (raw / "videos").iterdir()) == ["c0.mp4", "c1.mov"]
    assert not list(home.rglob("evil*"))
    job = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert job["kind"] == "dataset_prepare" and job["dataset_id"] == "custom"
    with open_session() as s:
        payload = s.get(Job, r.json()["id"]).payload
    assert payload["name"] == "mine" and payload["calibration"] == str(raw / "calibration.json")
    assert payload["gt"] == str(raw / "gt.csv") and len(payload["videos"]) == 2
    assert (
        client.post(
            "/api/datasets/custom", data={"name": "mine"}, files=[("videos", ("a.mp4", io.BytesIO(data), "video/mp4"))]
        ).status_code
        == 409
    )
    custom = next(d for d in client.get("/api/datasets").json() if d["id"] == "custom")
    assert custom["status"] == "preparing" and custom["job_id"] == r.json()["id"]


@pytest.mark.parametrize(
    "gt",
    [b"1,1,10,10,5,5,1,-1,-1,-1\n", b"frame,id,x,y,w,h\n", b"c3,0,1,1,1,5,5\n", b""],
    ids=["mot", "bad", "cam", "empty"],
)
def test_upload_custom_rejects_bad_gt_early(client, home, tmp_path, gt):
    r = client.post(
        "/api/datasets/custom",
        data={"name": "mine"},
        files=[
            ("videos", ("a.mp4", io.BytesIO(_mp4_bytes(tmp_path)), "video/mp4")),
            ("gt", ("gt.csv", io.BytesIO(gt))),
        ],
    )
    assert r.status_code == 422, r.text
    assert "camera_id,frame_id,person_id,x,y,w,h" in r.json()["detail"]
    assert not (paths().raw / "custom" / "mine").exists()


def test_upload_size_limit(client, home, tmp_path, monkeypatch):
    import mtmct.api.datasets as ds

    monkeypatch.setattr(ds, "MAX_VIDEO_BYTES", 100)
    r = client.post(
        "/api/datasets/custom",
        data={"name": "big"},
        files=[("videos", ("a.mp4", io.BytesIO(b"x" * 1000), "video/mp4"))],
    )
    assert r.status_code == 413 and not (paths().raw / "custom" / "big").exists()


def test_datasets_listing_and_fetch_job(client, seq, fake_adapter):
    ensure_media(seq)
    items = {d["id"]: d for d in client.get("/api/datasets").json()}
    assert items["custom"]["custom"] is True
    d = items["fakeds"]
    assert d["status"] == "absent" and d["licence"] == "MIT" and d["sequences"] == []
    r = client.post("/api/datasets/fakeds/fetch")
    assert r.status_code == 200 and r.json()["kind"] == "dataset_fetch"
    assert client.post("/api/datasets/fakeds/fetch").status_code == 409  # already queued
    assert client.post("/api/datasets/unknown/fetch").status_code == 404
    Worker(runner=FakeRunner()).tick()
    with open_session() as s:
        assert s.get(Job, r.json()["id"]).status == "done"
        assert s.get(Dataset, "fakeds").status == "ready"
    # toy/seq is only visible under its own dataset id; the fake adapter 'prepared' it
    assert (seq.root / "proxy" / "c0.mp4").exists()


def test_dataset_sequences_listed_when_prepared(client, seq, fake_adapter, monkeypatch):
    ADAPTERS["fakeds"].id = "toy"  # type: ignore[attr-defined]
    ADAPTERS["toy"] = ADAPTERS.pop("fakeds")
    ADAPTERS["toy"].meta = lambda: DatasetMeta(  # type: ignore[method-assign]
        id="toy",
        title="Toy",
        licence="MIT",
        source_url="",
        overlap="overlap",
        real=False,
        n_cameras="2",
        download_gb=0,
        sequences=["seq"],
    )
    try:
        ds = next(d for d in client.get("/api/datasets").json() if d["id"] == "toy")
        assert ds["status"] == "ready"
        s = ds["sequences"][0]
        assert s["name"] == "seq" and s["cameras"] == ["c0", "c1"] and s["duration_s"] == 2.0
        assert s["has_gt"] is False and s["has_calibration"] is False and s["media_ready"] is False
        assert s["splits"]["tune"] == [0.0, 0.5]
        ensure_media(seq)
        s = next(d for d in client.get("/api/datasets").json() if d["id"] == "toy")["sequences"][0]
        assert s["media_ready"] is True and s["preview_urls"]["c0"] == "/api/media/toy/seq/c0.jpg"
    finally:
        ADAPTERS.pop("toy", None)


def test_registry_lists_components_with_schema(client, home):
    from pydantic import BaseModel

    from mtmct import registry

    @registry.register("solver", "_api_test_solver")
    class _S:
        """Test solver."""

        LICENCE = "MIT"

        class Params(BaseModel):
            threshold: float = 0.4

        def __init__(self, params):
            self.params = params

    out = client.get("/api/registry").json()["components"]
    item = next(c for c in out["solver"] if c["name"] == "_api_test_solver")
    assert item["params_schema"]["properties"]["threshold"]["default"] == 0.4
    assert item["licence"] == "MIT" and item["category"] == "solver" and item["description"] == "Test solver."
    registry._REGISTRY["solver"].pop("_api_test_solver")


def test_profiles_and_system(client, home):
    profiles = {p["name"]: p for p in client.get("/api/profiles").json()}
    assert set(profiles) == {"overlap", "nonoverlap"}
    assert "geometry" in profiles["overlap"]["config"]["associate"]["costs"]
    assert "time" in profiles["nonoverlap"]["config"]["associate"]["costs"]
    s = client.get("/api/system").json()
    assert s["cpu_threads"] >= 1 and s["ram_total_gb"] > 0 and s["disk_free_gb"] > 0
    assert isinstance(s["openvino_devices"], list) and s["version"]
    assert all(item["kind"] in ("model", "dataset", "library") for item in s["licences"])


def test_cors_and_spa_fallback(home, tmp_path):
    from mtmct.api.app import create_app

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>spa</html>")
    (dist / "assets" / "a.js").write_text("console.log(1)")
    c = TestClient(create_app(static_dir=dist))
    assert c.get("/").text == "<html>spa</html>"
    assert c.get("/runs/12/player").text == "<html>spa</html>"
    assert c.get("/assets/a.js").text == "console.log(1)"
    assert c.get("/api/nope").status_code == 404  # API misses are not swallowed by the SPA
    r = c.options("/api/runs", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert (
        c.get("/api/runs", headers={"Origin": "http://evil.example"}).headers.get("access-control-allow-origin") is None
    )
