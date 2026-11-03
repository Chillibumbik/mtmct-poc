"""Heavy, cached stages (spec §8.2–8.5). Each writes its outputs into a stage directory and streams video."""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from mtmct import registry
from mtmct.components import FootPoint
from mtmct.config import RunConfig
from mtmct.data.base import CameraInfo, SequenceInfo
from mtmct.io import TRACKLET_ID_CAMERA_FACTOR, Detections, EmbeddingIndex, Tracklets, read_table, write_table
from mtmct.pipeline.events import Reporter
from mtmct.refine.quality import box_quality
from mtmct.refine.refine import refine
from mtmct.runtime.profiling import StageStats
from mtmct.video.reader import iter_frames

DETECT_BATCH = 8
REID_BATCH = 32
TRACK_CHECK_EVERY = 50  # frames between progress/cancel checks in the tracker loop
BOX_COLS = ["x", "y", "w", "h", "score"]

Frames = dict[str, np.ndarray]  # camera_id → processed frame ids
TRange = tuple[float, float]


def processed_fps(cfg: RunConfig, seq: SequenceInfo) -> float:
    """Rate of processed frames (target_fps never upsamples a camera)."""
    slowest = min(c.fps for c in seq.cameras)
    return slowest if cfg.data.target_fps <= 0 else min(cfg.data.target_fps, slowest)


def _rows(cam: CameraInfo, frame_ids: np.ndarray, boxes: np.ndarray) -> pd.DataFrame:
    """One camera's boxes `(n, ≥5)` = x, y, w, h, score → rows with camera_id, frame_id, ts."""
    df = pd.DataFrame(boxes[:, :5], columns=BOX_COLS)
    df.insert(0, "camera_id", cam.camera_id)
    df.insert(1, "frame_id", frame_ids.astype(np.int64))
    df.insert(2, "ts", np.asarray(cam.ts(frame_ids), dtype=np.float64))
    return df


def _concat(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _valid_boxes(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["w"] > 0) & (df["h"] > 0)]
    return df.assign(score=df["score"].clip(0, 1))


def detect(
    seq: SequenceInfo, cfg: RunConfig, t_range: TRange, frames: Frames, out: Path, rep: Reporter, st: StageStats
) -> None:
    detector = registry.create("detector", cfg.detect.name, cfg.detect.params)
    total, done, parts = max(sum(len(f) for f in frames.values()), 1), 0, []
    for cam in seq.cameras:
        fids, boxes = [np.zeros(0, np.int64)], [np.zeros((0, 5))]
        for batch in itertools.batched(iter_frames(seq, cam, cfg.data.target_fps, t_range), DETECT_BATCH):
            t0 = time.perf_counter()
            dets = detector.detect([f.image for f in batch])
            st.add(len(batch), time.perf_counter() - t0)
            for f, d in zip(batch, dets, strict=True):
                boxes.append(np.asarray(d, dtype=np.float64).reshape(-1, 5))
                fids.append(np.full(len(boxes[-1]), f.frame_id))
            done += len(batch)
            rep.progress(min(done / total, 1.0))
            rep.check()
        parts.append(_rows(cam, np.concatenate(fids), np.concatenate(boxes)))
    df = _valid_boxes(_concat(parts, list(Detections.to_schema().columns)))
    write_table(df, out / "detections.parquet", Detections)


def finish_tracklets(df: pd.DataFrame, seq: SequenceInfo, foot: FootPoint) -> pd.DataFrame:
    """Raw boxes (camera_id, frame_id, ts, tracklet_id, x, y, w, h, score) → Tracklets with foot and quality."""
    df = _valid_boxes(df).reset_index(drop=True)
    foot_xy = foot.foot(df[["x", "y", "w", "h"]].to_numpy(np.float64), None) if len(df) else np.zeros((0, 2))
    quality = np.zeros(len(df))
    for cam in seq.cameras:
        m = (df["camera_id"] == cam.camera_id).to_numpy()
        if m.any():
            quality[m] = box_quality(df[m], cam.width, cam.height)
    df = df.assign(
        quality=quality, foot_x=foot_xy[:, 0], foot_y=foot_xy[:, 1], world_x=np.nan, world_y=np.nan, emb_idx=-1
    )
    return Tracklets.validate(df[list(Tracklets.to_schema().columns)])


def _foot(cfg: RunConfig) -> FootPoint:
    return registry.create("foot_point", cfg.geometry.foot_point.name, cfg.geometry.foot_point.params)


def track(
    seq: SequenceInfo,
    cfg: RunConfig,
    t_range: TRange,
    frames: Frames,
    det_dir: Path,
    out: Path,
    rep: Reporter,
    st: StageStats,
) -> None:
    dets = read_table(det_dir / "detections.parquet", Detections)
    params = dict(cfg.track.params)
    if "frame_rate" in registry.get("tracker", cfg.track.name).Params.model_fields:
        params.setdefault("frame_rate", processed_fps(cfg, seq))
    tracker = registry.create("tracker", cfg.track.name, params)
    uses_frame = getattr(tracker, "USES_FRAME", True)  # unknown trackers get the frame to be safe
    total, done, parts = max(sum(len(f) for f in frames.values()), 1), 0, []
    no_image = np.zeros((0, 0, 3), np.uint8)
    for ci, cam in enumerate(seq.cameras):
        tracker.reset()
        fids, boxes, tids = [np.zeros(0, np.int64)], [np.zeros((0, 5))], [np.zeros(0, np.int64)]
        cam_dets = dets[dets["camera_id"] == cam.camera_id]
        det_boxes = cam_dets[BOX_COLS].to_numpy(np.float32)
        by_frame = {f: det_boxes[i] for f, i in cam_dets.groupby("frame_id").indices.items()}
        source: Iterator[tuple[int, np.ndarray]] = (
            ((f.frame_id, f.image) for f in iter_frames(seq, cam, cfg.data.target_fps, t_range))
            if uses_frame
            else ((int(f), no_image) for f in frames[cam.camera_id])
        )
        for frame_id, image in source:
            t0 = time.perf_counter()
            out_boxes = tracker.update(by_frame.get(frame_id, np.zeros((0, 5), np.float32)), image)
            st.add(1, time.perf_counter() - t0)
            if len(out_boxes):
                fids.append(np.full(len(out_boxes), frame_id))
                boxes.append(np.asarray(out_boxes[:, :5], np.float64))
                tids.append(ci * TRACKLET_ID_CAMERA_FACTOR + out_boxes[:, 5].astype(np.int64))
            done += 1
            if done % TRACK_CHECK_EVERY == 0:
                rep.progress(done / total)
                rep.check()
        part = _rows(cam, np.concatenate(fids), np.concatenate(boxes))
        part.insert(3, "tracklet_id", np.concatenate(tids))
        parts.append(part)
    raw = _concat(parts, ["camera_id", "frame_id", "ts", "tracklet_id", *BOX_COLS])
    write_table(finish_tracklets(raw, seq, _foot(cfg)), out / "tracklets.parquet", Tracklets)


def oracle_tracklets(seq: SequenceInfo, frames: Frames, foot: FootPoint, seed: int = 0) -> pd.DataFrame:
    """GT boxes on processed frames → Tracklets. Local ids are a per-camera random permutation of person ids,
    so tracklet ids carry no cross-camera identity."""
    gt = seq.gt()
    if gt is None:
        raise ValueError(f"oracle mode needs ground truth; {seq.dataset}/{seq.sequence} has none")
    rng = np.random.default_rng(seed)
    parts = []
    for ci, cam in enumerate(seq.cameras):
        g = gt[(gt["camera_id"] == cam.camera_id) & gt["frame_id"].isin(frames[cam.camera_id])]
        persons = np.unique(g["person_id"].to_numpy())
        local = dict(zip(persons.tolist(), rng.permutation(len(persons)).tolist(), strict=True))
        boxes = g[["x", "y", "w", "h"]].to_numpy(np.float64)
        part = _rows(cam, g["frame_id"].to_numpy(), np.column_stack([boxes, np.ones(len(g))]))
        part.insert(3, "tracklet_id", ci * TRACKLET_ID_CAMERA_FACTOR + g["person_id"].map(local).to_numpy(np.int64))
        parts.append(part)
    raw = _concat(parts, ["camera_id", "frame_id", "ts", "tracklet_id", *BOX_COLS])
    return finish_tracklets(raw, seq, foot)


def oracle(seq: SequenceInfo, cfg: RunConfig, frames: Frames, out: Path, st: StageStats) -> None:
    df = oracle_tracklets(seq, frames, _foot(cfg))
    st.items = len(df)
    write_table(df, out / "tracklets.parquet", Tracklets)


def reid_selection(trk: pd.DataFrame, stride: int, top_k: int) -> np.ndarray:
    """Boxes to embed: every `stride`-th box of each tracklet (in time) plus its top-k boxes by quality."""
    pos = trk.sort_values(["tracklet_id", "frame_id"]).groupby("tracklet_id").cumcount()
    by_quality = trk.sort_values("quality", ascending=False, kind="stable").groupby("tracklet_id").cumcount()
    return ((pos.reindex(trk.index) % max(stride, 1) == 0) | (by_quality.reindex(trk.index) < top_k)).to_numpy()


def _crop(image: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    h, w = image.shape[:2]
    x0, y0 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    x1, y1 = min(int(np.ceil(box[0] + box[2])), w), min(int(np.ceil(box[1] + box[3])), h)
    return image[y0:y1, x0:x1].copy() if x1 - x0 >= 2 and y1 - y0 >= 2 else None


def reid(
    seq: SequenceInfo, cfg: RunConfig, t_range: TRange, trk_dir: Path, out: Path, rep: Reporter, st: StageStats
) -> None:
    trk = read_table(trk_dir / "tracklets.parquet", Tracklets).reset_index(drop=True)
    selected = reid_selection(trk, cfg.portrait.reid_stride, cfg.portrait.top_k)
    model = registry.create("reid", cfg.reid.name, cfg.reid.params)
    total = max(int(selected.sum()), 1)
    embs: list[np.ndarray] = []
    owners: list[int] = []  # trk row of each embedding
    crops: list[np.ndarray] = []
    pending: list[int] = []

    def flush() -> None:
        if crops:
            t0 = time.perf_counter()
            embs.append(np.asarray(model.embed(crops), dtype=np.float16))
            st.add(len(crops), time.perf_counter() - t0)
            owners.extend(pending)
            crops.clear()
            pending.clear()
            rep.progress(len(owners) / total)
            rep.check()

    for cam in seq.cameras:
        need = trk[selected & (trk["camera_id"] == cam.camera_id).to_numpy()]
        if need.empty:
            continue
        rows_by_frame = {f: need.index.to_numpy()[i] for f, i in need.groupby("frame_id").indices.items()}
        last = int(need["frame_id"].max())
        for frame in iter_frames(seq, cam, cfg.data.target_fps, t_range):
            if frame.frame_id > last:
                break
            for row in rows_by_frame.get(frame.frame_id, ()):
                crop = _crop(frame.image, trk.loc[row, ["x", "y", "w", "h"]].to_numpy(np.float64))
                if crop is not None:
                    crops.append(crop)
                    pending.append(int(row))
            if len(crops) >= REID_BATCH:
                flush()
        flush()

    emb = np.concatenate(embs) if embs else np.zeros((0, model.dim), np.float16)
    trk.loc[owners, "emb_idx"] = np.arange(len(owners))
    np.save(out / "embeddings.npy", emb)
    write_table(trk, out / "tracklets.parquet", Tracklets)
    write_table(embedding_index(trk), out / "embeddings_index.parquet", EmbeddingIndex)


def embedding_index(trk: pd.DataFrame) -> pd.DataFrame:
    df = trk.loc[trk["emb_idx"] >= 0, ["emb_idx", "camera_id", "frame_id", "tracklet_id"]]
    return df.sort_values("emb_idx").reset_index(drop=True)


def refine_stage(cfg: RunConfig, fps: float, frames: Frames, reid_dir: Path, out: Path, st: StageStats) -> None:
    trk = read_table(reid_dir / "tracklets.parquet", Tracklets)
    emb = np.load(reid_dir / "embeddings.npy").astype(np.float32)
    t0 = time.perf_counter()
    refined = refine(trk, emb, cfg.refine, fps, frames)
    st.add(len(trk), time.perf_counter() - t0)
    write_table(refined, out / "tracklets.parquet", Tracklets)
    write_table(embedding_index(refined), out / "embeddings_index.parquet", EmbeddingIndex)
