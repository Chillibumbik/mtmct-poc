"""Proxy videos and preview frames (HTTP Range handled by FileResponse)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from mtmct.api.common import load_seq
from mtmct.worker.media import preview_path, proxy_path

router = APIRouter()


@router.get("/media/{dataset}/{sequence}/{filename}")
def media(dataset: str, sequence: str, filename: str) -> FileResponse:
    """`<camera>.mp4` proxy video or `<camera>.jpg` preview; the camera must exist in cameras.json."""
    stem, _, ext = filename.rpartition(".")
    if ext not in ("mp4", "jpg"):
        raise HTTPException(404, "unsupported media type")
    seq = load_seq(dataset, sequence)
    if stem not in {c.camera_id for c in seq.cameras}:
        raise HTTPException(404, f"camera {stem!r} not found")
    path = proxy_path(seq.root, stem) if ext == "mp4" else preview_path(seq.root, stem)
    if not path.exists():
        raise HTTPException(404, "media not generated yet")
    return FileResponse(path, media_type="video/mp4" if ext == "mp4" else "image/jpeg")
