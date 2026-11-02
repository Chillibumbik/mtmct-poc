"""Dataset catalogue, fetch jobs and custom video uploads (spec §10.1)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Annotated, BinaryIO

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select

from mtmct.api.common import SessionDep, job_out
from mtmct.api.schemas import DatasetOut, JobOut, SequenceOut
from mtmct.config import paths
from mtmct.data.base import DatasetMeta, list_sequences, load_adapters, load_sequence
from mtmct.db import Dataset, Job
from mtmct.worker.media import media_ready, preview_path

router = APIRouter()

MAX_VIDEO_BYTES = 2 * 1024**3
MAX_GT_BYTES = 200 * 1024**2
MAX_CALIBRATION_BYTES = 5 * 1024**2
MAX_CAMERAS = 16
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CUSTOM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

CUSTOM_META = DatasetMeta(
    id="custom",
    title="Собственные видео",
    licence="определяется пользователем",
    source_url="",
    overlap="mixed",
    real=True,
    n_cameras="1+",
    download_gb=0.0,
    sequences=[],
    description="Загруженные пользователем видео камер с опциональной калибровкой и разметкой.",
)


def _sequences(dataset_id: str) -> list[SequenceOut]:
    out = []
    for name in list_sequences(paths().data, dataset_id):
        try:
            seq = load_sequence(paths().data, dataset_id, name)
        except (FileNotFoundError, ValueError):
            continue
        out.append(
            SequenceOut(
                name=name,
                cameras=[c.camera_id for c in seq.cameras],
                duration_s=seq.meta.duration_s,
                splits=seq.meta.splits,
                has_gt=seq.has_gt,
                has_calibration=all(c.homography_img2world is not None for c in seq.cameras),
                preview_urls={
                    c.camera_id: f"/api/media/{dataset_id}/{name}/{c.camera_id}.jpg"
                    for c in seq.cameras
                    if preview_path(seq.root, c.camera_id).exists()
                },
                media_ready=media_ready(seq),
            )
        )
    return out


def _dataset_out(meta: DatasetMeta, row: Dataset | None, custom: bool = False) -> DatasetOut:
    sequences = _sequences(meta.id)
    status = row.status if row else "absent"
    if status in ("absent", "downloaded", "ready"):
        status = "ready" if sequences else "absent"
    return DatasetOut(
        id=meta.id,
        title=meta.title,
        licence=meta.licence,
        source_url=meta.source_url,
        overlap=meta.overlap,
        real=meta.real,
        n_cameras=meta.n_cameras,
        download_gb=meta.download_gb,
        requires_auth=meta.requires_auth,
        description=meta.description,
        status=status,  # type: ignore[arg-type]
        status_message=row.status_message if row else "",
        job_id=row.job_id if row else None,
        sequences=sequences,
        custom=custom,
    )


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(session: SessionDep) -> list[DatasetOut]:
    rows = {d.id: d for d in session.exec(select(Dataset)).all()}
    items = [_dataset_out(a.meta(), rows.get(a.id)) for a in load_adapters().values()]
    return [*items, _dataset_out(CUSTOM_META, rows.get("custom"), custom=True)]


def _has_active_job(session: Session, dataset_id: str) -> bool:
    stmt = select(Job).where(Job.dataset_id == dataset_id, Job.status.in_(("queued", "running")))  # type: ignore[attr-defined]
    return session.exec(stmt).first() is not None


def _enqueue_dataset_job(session: Session, kind: str, dataset_id: str, title: str, payload: dict) -> Job:
    job = Job(kind=kind, title=title, dataset_id=dataset_id, payload=payload)
    session.add(job)
    session.flush()
    row = session.get(Dataset, dataset_id) or Dataset(id=dataset_id)
    row.status = "downloading" if kind == "dataset_fetch" else "preparing"
    row.status_message, row.job_id = "в очереди", job.id
    session.add(row)
    session.commit()
    session.refresh(job)
    return job


@router.post("/datasets/{dataset_id}/fetch", response_model=JobOut)
def fetch_dataset(dataset_id: str, session: SessionDep) -> JobOut:
    adapter = load_adapters().get(dataset_id)
    if adapter is None:
        raise HTTPException(404, f"unknown dataset {dataset_id!r}")
    if _has_active_job(session, dataset_id):
        raise HTTPException(409, "a job for this dataset is already queued or running")
    title = f"Скачать и подготовить «{adapter.meta().title}»"
    return job_out(_enqueue_dataset_job(session, "dataset_fetch", dataset_id, title, {"dataset_id": dataset_id}))


def _save_limited(src: BinaryIO, dst: Path, limit: int) -> None:
    size = 0
    with dst.open("wb") as out:
        while chunk := src.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"file is larger than {limit // 1024**2} MB")
            out.write(chunk)


def _is_video(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.isOpened() and cap.read()[0]
    finally:
        cap.release()


@router.post("/datasets/custom", response_model=JobOut)
def upload_custom(
    name: Annotated[str, Form()],
    videos: Annotated[list[UploadFile], File()],
    session: SessionDep,
    calibration: Annotated[UploadFile | None, File()] = None,
    gt: Annotated[UploadFile | None, File()] = None,
) -> JobOut:
    """Store the upload under `data/_raw/custom/<name>/` with generated file names, then queue `dataset_prepare`."""
    if not CUSTOM_NAME.match(name):
        raise HTTPException(422, "name must be 1-64 chars of letters, digits, '_' or '-'")
    if not 1 <= len(videos) <= MAX_CAMERAS:
        raise HTTPException(422, f"expected 1..{MAX_CAMERAS} videos")
    raw = paths().raw / "custom" / name
    if raw.exists() or (paths().data / "custom" / name).exists():
        raise HTTPException(409, f"custom sequence {name!r} already exists")
    (raw / "videos").mkdir(parents=True)
    try:
        saved = []
        for i, up in enumerate(videos):
            ext = Path(up.filename or "").suffix.lower()  # only the extension is taken from the client
            dst = raw / "videos" / f"c{i}{ext if ext in VIDEO_EXTS else '.mp4'}"
            _save_limited(up.file, dst, MAX_VIDEO_BYTES)
            if not _is_video(dst):
                raise HTTPException(415, f"file #{i + 1} is not a readable video")
            saved.append(str(dst))
        calib_path = gt_path = None
        if calibration is not None:
            blob = calibration.file.read(MAX_CALIBRATION_BYTES + 1)
            try:
                if len(blob) > MAX_CALIBRATION_BYTES:
                    raise ValueError("too large")
                parsed = json.loads(blob)
            except ValueError as e:
                raise HTTPException(422, f"calibration must be a JSON file up to 5 MB: {e}") from e
            calib_path = raw / "calibration.json"
            calib_path.write_text(json.dumps(parsed))
        if gt is not None:
            gt_path = raw / "gt.csv"
            _save_limited(gt.file, gt_path, MAX_GT_BYTES)
        payload = {
            "dataset_id": "custom",
            "name": name,
            "videos": saved,
            "calibration": str(calib_path) if calib_path else None,
            "gt": str(gt_path) if gt_path else None,
        }
        return job_out(
            _enqueue_dataset_job(session, "dataset_prepare", "custom", f"Подготовка набора «{name}»", payload)
        )
    except BaseException:
        shutil.rmtree(raw, ignore_errors=True)
        raise
