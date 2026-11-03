"""Identity-level metrics. Clustering scores (spec §9.3) are over predicted tracklets: item = tracklet with a matched
box, true class = majority GT person of its matched boxes, cluster = global_id. Merge/split/handover use boxes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mtmct.eval.reid_sanity import tracklet_labels

MIN_BOXES, MIN_FRAC = 5, 0.1  # a person/id "owns" part of a cluster with ≥ max(5, 10 %) of its boxes


def _pairs(n: np.ndarray) -> float:
    return float((n * (n - 1) / 2).sum())


def cluster_scores(person: np.ndarray, cluster: np.ndarray) -> dict[str, float]:
    """Pairwise and BCubed P/R/F, purity, inverse purity of the clustering `cluster` against `person`."""
    n = len(person)
    if n == 0:
        keys = ("pairwise_p", "pairwise_r", "pairwise_f1", "bcubed_p", "bcubed_r", "bcubed_f", "purity")
        return dict.fromkeys((*keys, "inverse_purity"), 0.0)
    _, pi = np.unique(person, return_inverse=True)
    _, ci = np.unique(cluster, return_inverse=True)
    c = np.zeros((pi.max() + 1, ci.max() + 1))
    np.add.at(c, (pi, ci), 1)
    size_pred, size_gt = c.sum(0), c.sum(1)
    tp, pred_pairs, gt_pairs = _pairs(c), _pairs(size_pred), _pairs(size_gt)
    # no pairs claimed / no pairs to find: nothing can be wrong → 1
    pw_p = tp / pred_pairs if pred_pairs else 1.0
    pw_r = tp / gt_pairs if gt_pairs else 1.0
    bc_p = float((c**2 / size_pred[None, :]).sum() / n)
    bc_r = float((c**2 / size_gt[:, None]).sum() / n)
    return {
        "pairwise_p": pw_p,
        "pairwise_r": pw_r,
        "pairwise_f1": _f1(pw_p, pw_r),
        "bcubed_p": bc_p,
        "bcubed_r": bc_r,
        "bcubed_f": _f1(bc_p, bc_r),
        "purity": float(c.max(0).sum() / n),
        "inverse_purity": float(c.max(1).sum() / n),
    }


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _owners(counts: pd.Series) -> list[Any]:
    """Keys owning ≥ max(MIN_BOXES, MIN_FRAC · total) boxes, largest first (ties by key)."""
    thr = max(MIN_BOXES, MIN_FRAC * counts.sum())
    own = counts[counts >= thr]
    return [k for _, k in sorted(zip(-own.to_numpy(), own.index.tolist(), strict=True))]


def _case(m: pd.DataFrame, kind: str, gids: list[Any], persons: list[Any], first: pd.DataFrame) -> dict[str, Any]:
    involved = m[m["global_id"].isin(gids) & m["person_id"].isin(persons)]
    at = first.iloc[int(np.argmin(first["ts"].to_numpy()))]
    return {
        "kind": kind,
        "global_ids": gids,
        "gt_person_ids": persons,
        "tracklet_ids": sorted(involved["tracklet_id"].unique().tolist()),
        "camera_id": at["camera_id"],
        "frame_id": at["frame_id"],
        "t": at["ts"],
    }


def _errors(m: pd.DataFrame) -> tuple[int, int, list[dict[str, Any]]]:
    """False merges (id with ≥ 2 owning persons) and false splits (person with ≥ 2 owning ids), with UI cases.

    A case points at the first box of the second-largest owner, where the error becomes visible.
    """
    cells = m.groupby(["global_id", "person_id"]).size()
    cases, merges, splits = [], 0, 0
    for gid, counts in cells.groupby(level=0):
        persons = _owners(counts.droplevel(0))
        if len(persons) >= 2:
            merges += 1
            first = m[(m["global_id"] == gid) & (m["person_id"] == persons[1])]
            cases.append(_case(m, "false_merge", [gid], persons, first))
    for pid, counts in cells.groupby(level=1):
        gids = _owners(counts.droplevel(1))
        if len(gids) >= 2:
            splits += 1
            first = m[(m["person_id"] == pid) & (m["global_id"] == gids[1])]
            cases.append(_case(m, "false_split", gids, [pid], first))
    return merges, splits, cases


def _handover(m: pd.DataFrame) -> tuple[list[dict[str, Any]], float | None]:
    """Per GT person: cameras with matched boxes ordered by first appearance; each consecutive pair is a transition,
    kept if the person's majority global id in both cameras is equal (robust to simultaneous overlap views)."""
    if m.empty:
        return [], None
    first = m.groupby(["person_id", "camera_id"])["ts"].min().rename("t")
    cnt = m[["person_id", "camera_id", "global_id"]].value_counts().reset_index(name="n")
    cnt = cnt.sort_values(["person_id", "camera_id", "n", "global_id"], ascending=[True, True, False, True])
    maj = cnt.drop_duplicates(["person_id", "camera_id"]).set_index(["person_id", "camera_id"])["global_id"]
    seg = pd.concat([first, maj], axis=1).reset_index().sort_values(["person_id", "t", "camera_id"])
    person, cam, gid = seg["person_id"].to_numpy(), seg["camera_id"].to_numpy(), seg["global_id"].to_numpy()
    is_tr = person[1:] == person[:-1]
    tr = pd.DataFrame({"from_camera": cam[:-1][is_tr], "to_camera": cam[1:][is_tr]})
    tr["kept"] = (gid[:-1] == gid[1:])[is_tr]
    if tr.empty:
        return [], None
    agg = tr.groupby(["from_camera", "to_camera"])["kept"].agg(["size", "sum"]).reset_index()
    rows = [
        {"from_camera": a, "to_camera": b, "total": int(n), "kept": int(k)}
        for a, b, n, k in agg.itertuples(index=False)
    ]
    return rows, float(tr["kept"].mean())


def tracklet_cluster_scores(matches: pd.DataFrame) -> dict[str, float]:
    person = tracklet_labels(matches)
    gid = matches.groupby("tracklet_id")["global_id"].first()
    return cluster_scores(person.to_numpy(), gid.loc[person.index].to_numpy())


def identity_metrics(matches: pd.DataFrame, gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    """Identity metrics from matched boxes (`mot.matches_table`); `gt`/`pred` = rows inside the scored frames."""
    out: dict[str, Any] = tracklet_cluster_scores(matches)
    merges, splits, cases = _errors(matches)
    transitions, rate = _handover(matches)
    return out | {
        "n_pred_ids": int(pred["global_id"].nunique()),
        "n_gt_ids": int(gt["person_id"].nunique()),
        "false_merges": merges,
        "false_splits": splits,
        "handover_success": rate,
        "transitions": transitions,
        "cases": cases,
    }
