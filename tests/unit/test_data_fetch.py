from __future__ import annotations

import httpx

from mtmct.data.fetch import download, download_all

BODY = bytes(range(256)) * 40


def _handler(request: httpx.Request) -> httpx.Response:
    rng = request.headers.get("range")
    if rng:
        start = int(rng.removeprefix("bytes=").split("-")[0])
        return httpx.Response(
            206, content=BODY[start:], headers={"Content-Range": f"bytes {start}-{len(BODY) - 1}/{len(BODY)}"}
        )
    return httpx.Response(200, content=BODY, headers={"Content-Length": str(len(BODY))})


def _client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_handler))


def test_download_writes_file_atomically(tmp_path):
    dest = tmp_path / "a" / "f.bin"
    calls: list[float] = []
    out = download("http://x/f.bin", dest, lambda f, m: calls.append(f), client=_client())
    assert out == dest and dest.read_bytes() == BODY
    assert not dest.with_name("f.bin.part").exists()
    assert calls[-1] == 1.0


def test_download_resumes_part_file(tmp_path):
    dest = tmp_path / "f.bin"
    dest.with_name("f.bin.part").write_bytes(BODY[:1000])
    download("http://x/f.bin", dest, lambda f, m: None, client=_client())
    assert dest.read_bytes() == BODY


def test_download_skips_complete_file(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(BODY)

    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not hit the network")

    client = httpx.Client(transport=httpx.MockTransport(boom))
    assert download("http://x/f.bin", dest, lambda f, m: None, expected_size=len(BODY), client=client) == dest


def test_download_redownloads_when_size_differs(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"short")
    download("http://x/f.bin", dest, lambda f, m: None, expected_size=len(BODY), client=_client())
    assert dest.read_bytes() == BODY


def test_download_retries_dropped_connection_and_resumes(tmp_path):
    state = {"n": 0}

    class Dropping(httpx.SyncByteStream):
        def __iter__(self):
            yield BODY[:700]
            raise httpx.ReadError("connection dropped")

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, stream=Dropping(), headers={"Content-Length": str(len(BODY))})
        return _handler(request)

    dest = tmp_path / "f.bin"
    download("http://x/f.bin", dest, lambda f, m: None, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert dest.read_bytes() == BODY and state["n"] == 2


def test_download_all_reports_overall_progress(tmp_path):
    seen: list[float] = []
    jobs = [("http://x/a", tmp_path / "a", None), ("http://x/b", tmp_path / "b", None)]
    download_all(jobs, lambda f, m: seen.append(f), client=_client())
    assert (tmp_path / "a").exists() and (tmp_path / "b").exists()
    assert seen == sorted(seen) and seen[-1] == 1.0
