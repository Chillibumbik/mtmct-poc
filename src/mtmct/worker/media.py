"""Browser-friendly proxy videos and preview frames for prepared sequences."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import imageio_ffmpeg

from mtmct.data.base import CameraInfo, Progress, SequenceInfo
from mtmct.video.reader import IMAGE_EXTS

PREVIEW_SIDE = 480


def _first_frame(src: Path) -> cv2.typing.MatLike:
    if src.is_dir():
        first = next(p for p in sorted(src.iterdir()) if p.suffix.lower() in IMAGE_EXTS)
        image = cv2.imread(str(first), cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(str(src))
        ok, image = cap.read()
        cap.release()
        image = image if ok else None
    if image is None:
        raise ValueError(f"cannot read a frame from {src}")
    return image


def _even_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def make_proxy(src: Path, dst: Path, max_side: int = 960, fps: float | None = None) -> None:
    """H.264 / yuv420p / faststart copy with the long side <= max_side, source fps kept.

    `src` may be a directory of numbered frames (then `fps` is required).
    """
    h, w = _first_frame(src).shape[:2]
    out_w, out_h = _even_size(w, h, max_side)
    if src.is_dir():
        if fps is None:
            raise ValueError("fps is required for a directory of frames")
        suffix = next(p.suffix for p in sorted(src.iterdir()) if p.suffix.lower() in IMAGE_EXTS)
        inputs = ["-framerate", str(fps), "-pattern_type", "glob", "-i", str(src / f"*{suffix}")]
    else:
        inputs = ["-i", str(src)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.tmp.mp4")
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *inputs,
        "-vf",
        f"scale={out_w}:{out_h}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-g",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {src.name}: {proc.stderr.strip()[-500:]}")
    tmp.replace(dst)


def make_preview(src: Path, dst: Path, max_side: int = PREVIEW_SIDE) -> None:
    image = _first_frame(src)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 85])


def proxy_path(seq_root: Path, camera_id: str) -> Path:
    return seq_root / "proxy" / f"{camera_id}.mp4"


def preview_path(seq_root: Path, camera_id: str) -> Path:
    return seq_root / "preview" / f"{camera_id}.jpg"


def media_ready(seq: SequenceInfo) -> bool:
    return all(proxy_path(seq.root, c.camera_id).exists() for c in seq.cameras)


def _ensure_camera(seq: SequenceInfo, cam: CameraInfo) -> None:
    src = seq.root / cam.source
    if not proxy_path(seq.root, cam.camera_id).exists():
        make_proxy(src, proxy_path(seq.root, cam.camera_id), fps=cam.fps)
    if not preview_path(seq.root, cam.camera_id).exists():
        make_preview(src, preview_path(seq.root, cam.camera_id))


def ensure_media(
    seq: SequenceInfo, progress: Progress | None = None, should_cancel: Callable[[], bool] | None = None
) -> None:
    """Idempotently create proxies and previews for every camera of a sequence."""
    for i, cam in enumerate(seq.cameras):
        if should_cancel and should_cancel():
            return
        if progress:
            progress(i / len(seq.cameras), f"{seq.sequence}/{cam.camera_id}: proxy video")
        _ensure_camera(seq, cam)
    if progress:
        progress(1.0, f"{seq.sequence}: proxy videos ready")
