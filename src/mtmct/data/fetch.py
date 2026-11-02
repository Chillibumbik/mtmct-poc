"""Resumable streaming download: `<dest>.part` is extended with HTTP Range and renamed atomically."""

from __future__ import annotations

from pathlib import Path

import httpx

from mtmct.data.base import Progress

CHUNK = 1 << 20
RETRIES = 5


def download(
    url: str,
    dest: Path,
    progress: Progress,
    expected_size: int | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Download `url` to `dest`; a finished file (matching `expected_size` if given) is left alone."""
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        progress(1.0, f"{dest.name}: already downloaded")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0))
    try:
        for attempt in range(RETRIES):
            try:
                _stream(client, url, part, dest.name, progress)
                break
            except httpx.TransportError as e:  # dropped connection: resume from the .part file
                if attempt == RETRIES - 1:
                    raise
                progress(0.0, f"{dest.name}: {type(e).__name__}, retry {attempt + 1}")
    finally:
        if own:
            client.close()
    part.replace(dest)
    progress(1.0, f"{dest.name}: done")
    return dest


def _stream(client: httpx.Client, url: str, part: Path, name: str, progress: Progress) -> None:
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with client.stream("GET", url, headers=headers) as r:
        if r.status_code == 416:  # the .part file already holds the whole body
            return
        r.raise_for_status()
        if done and r.status_code != 206:  # server ignored Range: start over
            done = 0
        total = done + int(r.headers.get("Content-Length", 0))
        with part.open("ab" if done else "wb") as f:
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                done += len(chunk)
                progress(min(done / total, 0.999) if total else 0.0, f"{name}: {done / 1e6:.0f} MB")


def download_all(
    jobs: list[tuple[str, Path, int | None]], progress: Progress, client: httpx.Client | None = None
) -> None:
    """Download (url, dest, expected_size) jobs in order, reporting overall progress."""
    for i, (url, dest, size) in enumerate(jobs):

        def part(fraction: float, message: str, i: int = i) -> None:
            progress((i + fraction) / len(jobs), message)

        download(url, dest, part, expected_size=size, client=client)
