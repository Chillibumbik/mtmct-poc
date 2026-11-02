"""Tracklet clean-up (spec §8.5): split on appearance jumps, stitch fragments, drop junk, interpolate gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtmct.config import RefineCfg

_TAIL_ROWS = 5  # rows used to estimate the end velocity of a tracklet


def refine(tracklets: pd.DataFrame, emb: np.ndarray, cfg: RefineCfg, fps: float) -> pd.DataFrame:
    """Tracklets in, Tracklets out; `emb[emb_idx]` is a box embedding, `fps` the processed frame rate."""
    if not cfg.enabled or tracklets.empty:
        return tracklets
    df = tracklets.sort_values(["camera_id", "tracklet_id", "frame_id"]).reset_index(drop=True)
    df = _split(df, emb, cfg)
    df = _stitch(df, emb, cfg, fps)
    df = _drop(df, cfg)
    return _interpolate(df, cfg)


def _cuts(e: np.ndarray, window: int, thr: float) -> list[int]:
    """Indices `i` where mean(e[i-window:i]) and mean(e[i:i+window]) have cosine < thr (worst first, recursive)."""
    n = len(e)
    if n < 2 * window:
        return []
    cs = np.vstack([np.zeros((1, e.shape[1])), np.cumsum(e, axis=0)])
    i = np.arange(window, n - window + 1)
    prev, nxt = cs[i] - cs[i - window], cs[i + window] - cs[i]
    cos = (prev * nxt).sum(1) / np.maximum(np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1), 1e-12)
    k = int(np.argmin(cos))
    if cos[k] >= thr:
        return []
    cut = int(i[k])
    return [*_cuts(e[:cut], window, thr), cut, *(cut + c for c in _cuts(e[cut:], window, thr))]


def _split(df: pd.DataFrame, emb: np.ndarray, cfg: RefineCfg) -> pd.DataFrame:
    new_id = df["tracklet_id"].to_numpy().copy()
    next_id = df.groupby("camera_id")["tracklet_id"].max().to_dict()
    for rows in df.groupby("tracklet_id", sort=False).indices.values():
        emb_idx = df["emb_idx"].to_numpy()[rows]
        pos = np.flatnonzero(emb_idx >= 0)  # `rows` are in frame order
        cuts = _cuts(emb[emb_idx[pos]], cfg.window, cfg.split_thr)
        if not cuts:
            continue
        cam = df["camera_id"].iat[rows[0]]
        bounds = [(pos[c - 1] + pos[c]) // 2 + 1 for c in cuts]  # first row of each new part
        for part in np.split(rows, bounds)[1:]:
            next_id[cam] += 1
            new_id[part] = next_id[cam]
    return df.assign(tracklet_id=new_id)


def _frame_step(frame_id: pd.Series) -> int:
    d = np.diff(np.unique(frame_id.to_numpy()))
    return max(int(np.median(d)), 1) if len(d) else 1


def _stitch(df: pd.DataFrame, emb: np.ndarray, cfg: RefineCfg, fps: float) -> pd.DataFrame:
    mapping: dict[int, int] = {}
    for _, cam in df.groupby("camera_id", sort=False):
        mapping.update(_stitch_camera(cam, emb, cfg, fps))
    return df.assign(tracklet_id=df["tracklet_id"].map(mapping).fillna(df["tracklet_id"]).astype("int64"))


def _stitch_camera(cam: pd.DataFrame, emb: np.ndarray, cfg: RefineCfg, fps: float) -> dict[int, int]:
    """Greedy chaining of A-end → B-start by best score; returns tracklet_id → id of the first tracklet of its chain."""
    step = _frame_step(cam["frame_id"])
    ids: list[int] = []
    start, end, head, tail, vel, esum, cnt = [], [], [], [], [], [], []
    for _, g in cam.groupby("tracklet_id", sort=False):
        f = g["frame_id"].to_numpy()
        foot = g[["foot_x", "foot_y"]].to_numpy()
        k = max(len(g) - _TAIL_ROWS, 0)
        e = g["emb_idx"].to_numpy()
        e = emb[e[e >= 0]]
        ids.append(int(g["tracklet_id"].to_numpy()[0]))
        start.append(f[0])
        end.append(f[-1])
        head.append(foot[0])
        tail.append(foot[-1])
        vel.append((foot[-1] - foot[k]) / ((f[-1] - f[k]) / step) if f[-1] > f[k] else np.zeros(2))
        esum.append(e.sum(0) if len(e) else np.zeros(emb.shape[1]))
        cnt.append(len(e))
    n = len(ids)
    if n < 2:
        return {}
    start_a, end_a = np.array(start, float), np.array(end, float)
    head_a, tail_a, vel_a = np.array(head), np.array(tail), np.array(vel)
    esum_a, cnt_a = np.array(esum), np.array(cnt)
    active = np.ones(n, bool)

    def scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Score of chaining every chain in `a` (earlier) into every chain in `b` (later); -inf if not allowed."""
        gap = (start_a[b][None, :] - end_a[a][:, None]) / step  # processed frames
        pred = tail_a[a][:, None, :] + vel_a[a][:, None, :] * gap[..., None]
        dist = np.linalg.norm(pred - head_a[b][None, :, :], axis=2)
        ma, mb = esum_a[a], esum_a[b]
        na = np.maximum(np.linalg.norm(ma, axis=1), 1e-12)
        nb = np.maximum(np.linalg.norm(mb, axis=1), 1e-12)
        cos = (ma @ mb.T) / (na[:, None] * nb[None, :])
        ok = (gap > 0) & (gap / fps <= cfg.max_gap_s) & (dist <= cfg.stitch_dist_px) & (cos >= cfg.stitch_app_thr)
        ok &= (cnt_a[a] > 0)[:, None] & (cnt_a[b] > 0)[None, :] & active[a][:, None] & active[b][None, :]
        ok &= a[:, None] != b[None, :]
        return np.where(ok, cos - dist / cfg.stitch_dist_px, -np.inf)

    everyone = np.arange(n)
    s = scores(everyone, everyone)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    while np.isfinite(s.max()):
        i, j = (int(v) for v in np.unravel_index(np.argmax(s), s.shape))
        end_a[i], tail_a[i], vel_a[i] = end_a[j], tail_a[j], vel_a[j]
        esum_a[i], cnt_a[i] = esum_a[i] + esum_a[j], cnt_a[i] + cnt_a[j]
        members[i] += members.pop(j)
        active[j] = False
        s[j, :], s[:, j] = -np.inf, -np.inf
        s[i, :], s[:, i] = scores(np.array([i]), everyone)[0], scores(everyone, np.array([i]))[:, 0]
    return {ids[m]: ids[head_slot] for head_slot, ms in members.items() for m in ms}


def _drop(df: pd.DataFrame, cfg: RefineCfg) -> pd.DataFrame:
    g = df.groupby("tracklet_id")["score"]
    keep = (g.transform("size") >= cfg.min_len) & (g.transform("mean") >= cfg.min_score)
    return df[keep].reset_index(drop=True)


def _interpolate(df: pd.DataFrame, cfg: RefineCfg) -> pd.DataFrame:
    """Fill gaps of 1..interp_gap missing frames inside a tracklet; the frame period is inferred per camera."""
    df = df.sort_values(["camera_id", "tracklet_id", "frame_id"]).reset_index(drop=True)
    step = df["camera_id"].map(df.groupby("camera_id")["frame_id"].agg(_frame_step)).to_numpy()
    same = (df["tracklet_id"] == df["tracklet_id"].shift()).to_numpy()
    n_missing = np.zeros(len(df), dtype=int)
    n_missing[1:] = np.rint(np.diff(df["frame_id"].to_numpy()) / step[1:]).astype(int) - 1
    n_missing = np.where(same & (n_missing >= 1) & (n_missing <= cfg.interp_gap), n_missing, 0)
    if not n_missing.any():
        return df

    b = np.repeat(np.flatnonzero(n_missing), n_missing[n_missing > 0])  # row after the gap, once per new frame
    k = np.concatenate([np.arange(1, m + 1) for m in n_missing[n_missing > 0]])
    a = b - 1
    t = (k / (n_missing[b] + 1))[:, None]
    num = ["ts", "x", "y", "w", "h", "foot_x", "foot_y", "world_x", "world_y"]
    new = df.iloc[a].reset_index(drop=True)
    new[num] = df[num].to_numpy()[a] * (1 - t) + df[num].to_numpy()[b] * t
    new["frame_id"] = df["frame_id"].to_numpy()[a] + k * step[a]
    new["score"] = np.minimum(df["score"].to_numpy()[a], df["score"].to_numpy()[b])
    new["quality"] = 0.0
    new["emb_idx"] = -1
    out = pd.concat([df, new], ignore_index=True)
    return out.sort_values(["camera_id", "tracklet_id", "frame_id"]).reset_index(drop=True)
