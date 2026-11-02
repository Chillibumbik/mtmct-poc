"""Result views for the player, BEV, person routes and error browser (spec §10.1)."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

from mtmct.api.common import SessionDep, get_run, load_seq, run_dir
from mtmct.api.runs import read_json_result
from mtmct.api.schemas import (
    Box,
    ErrorCase,
    ErrorsOut,
    FrameBoxes,
    PersonRoute,
    RouteSegment,
    TrackBoxes,
    TrackletCrop,
    TrajectoriesOut,
    Trajectory,
    TrajectoryPoint,
    TransitionCell,
)
from mtmct.data.base import CameraInfo, SequenceInfo
from mtmct.io import GlobalTracks, GroundTruth, read_table
from mtmct.video.reader import IMAGE_EXTS

router = APIRouter()

MAX_FRAMES_PER_CALL = 300
CROPS_PER_TRACKLET = 4
CROP_HEIGHT = 128
TRAJECTORY_BUCKET_S = 0.5
MAX_ERROR_CASES = 100
MAX_TRACKLETS_PER_CASE = 8


@lru_cache(maxsize=8)
def _read_tracks(path: str, mtime_ns: int) -> pd.DataFrame:
    return read_table(Path(path), GlobalTracks)


@lru_cache(maxsize=4)
def _read_gt(path: str, mtime_ns: int) -> pd.DataFrame:
    return read_table(Path(path), GroundTruth)


def load_tracks(run_id: int) -> pd.DataFrame:
    path = run_dir(run_id) / "global_tracks.csv"
    if not path.exists():
        raise HTTPException(404, f"run {run_id} has no global_tracks.csv (not finished?)")
    return _read_tracks(str(path), path.stat().st_mtime_ns)


def load_gt(seq: SequenceInfo) -> pd.DataFrame | None:
    path = seq.root / "gt.parquet"
    return _read_gt(str(path), path.stat().st_mtime_ns) if path.exists() else None


def _camera(seq: SequenceInfo, camera_id: str) -> CameraInfo:
    try:
        return seq.camera(camera_id)
    except KeyError:
        raise HTTPException(404, f"camera {camera_id!r} not in {seq.dataset}/{seq.sequence}") from None


def _boxes(df: pd.DataFrame, id_col: str) -> dict[int, list[Box]]:
    out: dict[int, list[Box]] = {}
    has_tracklet = "tracklet_id" in df.columns
    for r in df.to_dict("records"):
        out.setdefault(int(r["frame_id"]), []).append(
            Box(
                x=r["x"],
                y=r["y"],
                w=r["w"],
                h=r["h"],
                id=int(r[id_col]),
                tracklet_id=int(r["tracklet_id"]) if has_tracklet else None,
            )
        )
    return out


@router.get("/runs/{run_id}/tracks", response_model=TrackBoxes)
def tracks(
    run_id: int,
    camera: str,
    session: SessionDep,
    t0: float = 0.0,
    t1: Annotated[float | None, Query(description="default: t0 + 30 s")] = None,
) -> TrackBoxes:
    """Boxes of one camera in [t0, t1) on the common clock; at most 300 frames per call (page by t0)."""
    run = get_run(session, run_id)
    seq = load_seq(run.dataset, run.sequence)
    cam = _camera(seq, camera)
    end = t0 + 30.0 if t1 is None else t1

    def window(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.camera_id == camera]
        ts = (df.frame_id + cam.frame_offset) / cam.fps
        return df[(ts >= t0 - 1e-9) & (ts < end - 1e-9)]

    pred = window(load_tracks(run_id))
    gt_all = load_gt(seq)
    gt = window(gt_all) if gt_all is not None else pd.DataFrame(columns=["frame_id"])
    frame_ids = sorted(set(pred.frame_id) | set(gt.frame_id))[:MAX_FRAMES_PER_CALL]
    pred_by, gt_by = _boxes(pred, "global_id"), (_boxes(gt, "person_id") if len(gt) else {})
    frames = [
        FrameBoxes(frame_id=int(f), ts=float(cam.ts(int(f))), pred=pred_by.get(int(f), []), gt=gt_by.get(int(f), []))
        for f in frame_ids
    ]
    return TrackBoxes(
        camera_id=camera, fps=cam.fps, frame_offset=cam.frame_offset, width=cam.width, height=cam.height, frames=frames
    )


@router.get("/runs/{run_id}/trajectories", response_model=TrajectoriesOut)
def trajectories(run_id: int, session: SessionDep) -> TrajectoriesOut:
    run = get_run(session, run_id)
    seq = load_seq(run.dataset, run.sequence)
    df = load_tracks(run_id)
    df = df[df.world_x.notna() & df.world_y.notna()]
    if df.empty:
        return TrajectoriesOut(available=False, bounds=None, trajectories=[])
    fps = df.camera_id.map({c.camera_id: c.fps for c in seq.cameras})
    offset = df.camera_id.map({c.camera_id: c.frame_offset for c in seq.cameras})
    df = df.assign(ts=(df.frame_id + offset) / fps)
    df = df.assign(bucket=(df.ts / TRAJECTORY_BUCKET_S).astype(int))
    # one point per person per time bucket (overlapping cameras see the same person at the same instant)
    df = df.sort_values(["global_id", "ts"]).drop_duplicates(["global_id", "bucket"])
    trajs = [
        Trajectory(
            global_id=int(gid),
            points=[
                TrajectoryPoint(ts=r["ts"], x=r["world_x"], y=r["world_y"], camera_id=r["camera_id"])
                for r in df[df.global_id == gid].to_dict("records")
            ],
        )
        for gid in sorted(df.global_id.unique())
    ]
    bounds = (float(df.world_x.min()), float(df.world_y.min()), float(df.world_x.max()), float(df.world_y.max()))
    return TrajectoriesOut(available=True, bounds=bounds, trajectories=trajs)


def tracklet_gt_person(rows: pd.DataFrame, gt: pd.DataFrame | None) -> int | None:
    """GT person that overlaps the tracklet's boxes (IoU >= 0.5) most often, if GT exists."""
    if gt is None or rows.empty:
        return None
    m = rows.merge(gt, on=["camera_id", "frame_id"], suffixes=("", "_gt"))
    if m.empty:
        return None
    iw = np.minimum(m.x + m.w, m.x_gt + m.w_gt) - np.maximum(m.x, m.x_gt)
    ih = np.minimum(m.y + m.h, m.y_gt + m.h_gt) - np.maximum(m.y, m.y_gt)
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    iou = inter / (m.w * m.h + m.w_gt * m.h_gt - inter)
    hits = m.loc[iou >= 0.5, "person_id"]
    return int(hits.mode().iloc[0]) if len(hits) else None


def _crop_urls(run_id: int, tracklet_id: int, n_boxes: int) -> list[str]:
    return [f"/api/runs/{run_id}/crops/{tracklet_id}?i={i}" for i in range(min(CROPS_PER_TRACKLET, n_boxes))]


def _tracklet_span(rows: pd.DataFrame, cam: CameraInfo) -> tuple[float, float]:
    return float(cam.ts(int(rows.frame_id.min()))), float(cam.ts(int(rows.frame_id.max())))


@router.get("/runs/{run_id}/person/{global_id}", response_model=PersonRoute)
def person(run_id: int, global_id: int, session: SessionDep) -> PersonRoute:
    run = get_run(session, run_id)
    seq = load_seq(run.dataset, run.sequence)
    rows = load_tracks(run_id)
    rows = rows[rows.global_id == global_id]
    if rows.empty:
        raise HTTPException(404, f"global id {global_id} not in run {run_id}")
    gt = load_gt(seq)
    segments, gt_ids = [], set()
    for tid in rows.tracklet_id.unique():
        g = rows[rows.tracklet_id == tid]
        cam = seq.camera(str(g.camera_id.iloc[0]))
        t_start, t_end = _tracklet_span(g, cam)
        segments.append(
            RouteSegment(
                camera_id=cam.camera_id,
                t_start=t_start,
                t_end=t_end,
                tracklet_ids=[int(tid)],
                crop_url=_crop_urls(run_id, int(tid), len(g))[0],
            )
        )
        if (pid := tracklet_gt_person(g, gt)) is not None:
            gt_ids.add(pid)
    segments.sort(key=lambda s: s.t_start)
    return PersonRoute(global_id=global_id, segments=segments, gt_person_ids=sorted(gt_ids))


def _case_tracklets(
    run_id: int, seq: SequenceInfo, df: pd.DataFrame, gt: pd.DataFrame | None, case: dict[str, Any]
) -> list[TrackletCrop]:
    ids: Iterable[int] = (
        case.get("tracklet_ids") or df[df.global_id.isin(case.get("global_ids", []))].tracklet_id.unique()
    )
    out = []
    for tid in list(ids)[:MAX_TRACKLETS_PER_CASE]:
        g = df[df.tracklet_id == tid]
        if g.empty:
            continue
        cam = seq.camera(str(g.camera_id.iloc[0]))
        t_start, t_end = _tracklet_span(g, cam)
        out.append(
            TrackletCrop(
                tracklet_id=int(tid),
                camera_id=cam.camera_id,
                t_start=t_start,
                t_end=t_end,
                crop_urls=_crop_urls(run_id, int(tid), len(g)),
                gt_person_id=tracklet_gt_person(g, gt),
                global_id=int(g.global_id.iloc[0]),
            )
        )
    return out


@router.get("/runs/{run_id}/errors", response_model=ErrorsOut)
def errors(run_id: int, session: SessionDep) -> ErrorsOut:
    """False merges / splits from metrics.json `identity.cases`, enriched with tracklet crops."""
    run = get_run(session, run_id)
    seq = load_seq(run.dataset, run.sequence)
    identity = read_json_result(run_id, "metrics.json").get("identity", {})
    df, gt = load_tracks(run_id), load_gt(seq)
    result: dict[str, list[ErrorCase]] = {"false_merge": [], "false_split": []}
    for case in identity.get("cases", []):
        kind = case.get("kind")
        if kind not in result or len(result[kind]) >= MAX_ERROR_CASES:
            continue
        tracklets = _case_tracklets(run_id, seq, df, gt, case)
        if not tracklets:
            continue
        result[kind].append(
            ErrorCase(
                kind=kind,
                global_ids=case.get("global_ids", []),
                gt_person_ids=case.get("gt_person_ids", []),
                tracklets=tracklets,
                t=float(case.get("t", tracklets[0].t_start)),
                camera_id=case.get("camera_id", tracklets[0].camera_id),
            )
        )
    transitions = [TransitionCell(**t) for t in identity.get("transitions", [])]
    return ErrorsOut(false_merges=result["false_merge"], false_splits=result["false_split"], transitions=transitions)


def read_frame(seq: SequenceInfo, cam: CameraInfo, frame_id: int) -> np.ndarray | None:
    src = seq.root / cam.source
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return cv2.imread(str(files[frame_id]), cv2.IMREAD_COLOR) if frame_id < len(files) else None
    cap = cv2.VideoCapture(str(src))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = cap.read()
        return image if ok else None
    finally:
        cap.release()


@router.get("/runs/{run_id}/crops/{tracklet_id}")
def crop(run_id: int, tracklet_id: int, session: SessionDep, i: Annotated[int, Query(ge=0)] = 0) -> Response:
    """JPEG thumbnail (128 px high) of the tracklet at its i-th of up to 4 evenly spread boxes."""
    run = get_run(session, run_id)
    rows = load_tracks(run_id)
    rows = rows[rows.tracklet_id == tracklet_id].sort_values("frame_id")
    n = min(CROPS_PER_TRACKLET, len(rows))
    if i >= n:
        raise HTTPException(404, f"tracklet {tracklet_id} has {n} crops")
    cached = run_dir(run_id) / "crops" / f"{tracklet_id}_{i}.jpg"
    headers = {"Cache-Control": "private, max-age=300"}
    if cached.exists():
        return FileResponse(cached, media_type="image/jpeg", headers=headers)
    box = rows.iloc[min(len(rows) - 1, int((i + 0.5) / n * len(rows)))]
    seq = load_seq(run.dataset, run.sequence)
    cam = _camera(seq, str(box.camera_id))
    image = read_frame(seq, cam, int(box.frame_id))
    if image is None:
        raise HTTPException(404, "frame cannot be decoded")
    h, w = image.shape[:2]
    x0, y0 = max(0, round(box.x)), max(0, round(box.y))
    x1, y1 = min(w, round(box.x + box.w)), min(h, round(box.y + box.h))
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(404, "box is outside the frame")
    patch = image[y0:y1, x0:x1]
    out_w = min(2 * CROP_HEIGHT, max(1, round(patch.shape[1] * CROP_HEIGHT / patch.shape[0])))
    patch = cv2.resize(patch, (out_w, CROP_HEIGHT), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "jpeg encoding failed")
    data = buf.tobytes()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return Response(data, media_type="image/jpeg", headers=headers)
