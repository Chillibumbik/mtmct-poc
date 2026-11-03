"""Fine-tune OSNet-AIN x1_0 (MSMT17 init) on GT crops of the tune splits and export `models/osnet_ain_x1_0_ft.onnx`.

    uv run --group export python scripts/finetune_reid.py build    # crops → data/_ft/reid/ (index.csv)
    uv run --group export python scripts/finetune_reid.py train    # → models/_src/osnet_ain_x1_0_ft.pth, log
    uv run --group export python scripts/finetune_reid.py export   # → models/osnet_ain_x1_0_ft.onnx

Only `mtmct.data.finetune.TUNE_SOURCES` feeds training (see that module). Identity = (dataset, person_id);
~15 % of identities of each dataset are held out as validation (cross-camera rank-1 / mAP, never trained on).
Recipe: softmax CE (label smoothing 0.1) + batch-hard triplet (margin 0.3), P×K = 16×4, 256×128, flip, pad-crop,
colour jitter, random erasing; Adam, cosine LR, classifier-only first epoch. Stops when val mAP stops improving.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from export_osnet import load_factory, to_onnx

from mtmct.config import paths
from mtmct.data.base import load_sequence
from mtmct.data.finetune import TUNE_SOURCES, crop, read_frames, sample_reid_boxes, tune_gt
from mtmct.eval.reid_sanity import reid_sanity
from mtmct.models import MODELS, fetch_source

H, W = 256, 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ROOT = paths().data / "_ft" / "reid"
CKPT = paths().models / "_src" / "osnet_ain_x1_0_ft.pth"
LAST = CKPT.with_suffix(".last.pth")  # resume state after every epoch
VAL_MOD = 7  # identities with crc32(dataset/person) % 7 == 0 → validation (~14 %)


def build(per_camera: int) -> None:
    rows = []
    for (ds, name), split in TUNE_SOURCES.items():
        seq = load_sequence(paths().data, ds, name)
        gt = tune_gt(seq, split)
        sizes = {c.camera_id: (c.width, c.height) for c in seq.cameras}
        picks = sample_reid_boxes(gt, per_camera=per_camera, sizes=sizes)
        out = ROOT / ds
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        for cam_id, boxes in picks.groupby("camera_id"):
            by_frame = boxes.groupby("frame_id")
            for fid, image in read_frames(seq, seq.camera(str(cam_id)), list(by_frame.groups)):
                for b in by_frame.get_group(fid).itertuples():
                    c = crop(image, b.x, b.y, b.w, b.h)
                    if c is None:
                        continue
                    rel = f"{ds}/{b.person_id}_{cam_id}_{fid}.jpg"
                    cv2.imwrite(str(ROOT / rel), cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA))
                    rows.append({"path": rel, "dataset": ds, "person_id": int(b.person_id), "camera_id": str(cam_id),
                                 "frame_id": int(fid), "ts": float(b.ts)})  # fmt: skip
        print(f"{ds}/{name} split={split}: {len(picks)} boxes, {picks.person_id.nunique()} ids, {time.time() - t:.0f}s")
    df = pd.DataFrame(rows)
    key = df.dataset + "/" + df.person_id.astype(str)
    df["val"] = [zlib.crc32(k.encode()) % VAL_MOD == 0 for k in key]
    df.to_csv(ROOT / "index.csv", index=False)
    print(df.groupby(["dataset", "val"]).agg(crops=("path", "size"), ids=("person_id", "nunique")))


def load_image(rel: str) -> np.ndarray:
    return cv2.imread(str(ROOT / rel), cv2.IMREAD_COLOR)


def to_tensor(images: list[np.ndarray]) -> torch.Tensor:
    x = (np.stack(images)[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    # channels_last: depthwise convs of OSNet run ~2x faster on CPU
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        img = img[:, ::-1]
    img = cv2.copyMakeBorder(np.ascontiguousarray(img), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    y, x = rng.integers(0, 21, 2)
    img = img[y : y + H, x : x + W].astype(np.float32)
    # colour jitter: brightness, contrast, saturation ±20 %
    img = img * rng.uniform(0.8, 1.2)
    img = (img - img.mean()) * rng.uniform(0.8, 1.2) + img.mean()
    gray = img.mean(axis=2, keepdims=True)
    img = np.clip((img - gray) * rng.uniform(0.8, 1.2) + gray, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:  # random erasing (area 2-40 %, aspect 0.3-3.3), filled with noise
        for _ in range(10):
            area = rng.uniform(0.02, 0.4) * H * W
            r = np.exp(rng.uniform(np.log(0.3), np.log(3.3)))
            h, w = round(np.sqrt(area * r)), round(np.sqrt(area / r))
            if h < H and w < W:
                y, x = rng.integers(0, H - h), rng.integers(0, W - w)
                img[y : y + h, x : x + w] = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                break
    return img


def triplet_hard(feat: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    d = torch.cdist(feat, feat)
    same = labels[:, None] == labels[None, :]
    hardest_pos = (d * same).max(1).values
    hardest_neg = (d + 1e6 * same).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


@torch.no_grad()
def embed(model: torch.nn.Module, paths_: list[str], batch: int = 128) -> np.ndarray:
    dev = next(model.parameters()).device
    model.eval()
    out = [
        model(to_tensor([load_image(p) for p in paths_[i : i + batch]]).to(dev)).cpu().numpy()
        for i in range(0, len(paths_), batch)
    ]
    return np.concatenate(out)


def validate(model: torch.nn.Module, val: pd.DataFrame) -> dict:
    emb = embed(model, val.path.tolist())
    labels = (val.dataset + "/" + val.person_id.astype(str)).to_numpy()
    cams = (val.dataset + "/" + val.camera_id).to_numpy()
    res = {"all": reid_sanity(emb, labels, cams)}
    for ds in sorted(val.dataset.unique()):
        m = (val.dataset == ds).to_numpy()
        res[ds] = reid_sanity(emb[m], labels[m], cams[m])
    return res


def load_init(num_classes: int) -> torch.nn.Module:
    state = torch.load(fetch_source("osnet_ain_x1_0"), map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.get("state_dict", state).items()}
    model = load_factory("osnet_ain_x1_0")(num_classes=num_classes, pretrained=False, loss="triplet")
    # the MSMT17 checkpoint holds denormal floats: on CPU they make every conv ~8x slower
    state = {k: torch.where(v.abs() < 1e-30, 0, v) for k, v in state.items() if not k.startswith("classifier.")}
    missing, _ = model.load_state_dict(state, strict=False)
    assert all(k.startswith("classifier.") for k in missing), missing
    return model


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    df = pd.read_csv(ROOT / "index.csv")
    tr = df[~df.val].reset_index(drop=True)
    # ≤ 10 crops per held-out identity per camera keeps validation under a minute
    val = df[df.val].groupby(["dataset", "person_id", "camera_id"]).head(10).reset_index(drop=True)
    key = tr.dataset + "/" + tr.person_id.astype(str)
    ids = sorted(key.unique())
    tr["label"] = key.map({k: i for i, k in enumerate(ids)})
    by_id = [g.index.to_numpy() for _, g in tr.groupby("label")]
    model = load_init(len(ids)).to(args.device, memory_format=torch.channels_last)
    resume = torch.load(LAST, map_location=args.device, weights_only=False) if LAST.exists() else None
    if resume is None:
        best = validate(model, val)
        print(f"epoch 0 (MSMT17 init) val {json.dumps(best['all'])}", flush=True)
        log = [{"epoch": 0, "val": best}]
        best_map, stale, start = best["all"]["mAP"], 0, 1
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CKPT)
    else:
        model.load_state_dict(resume["model"])
        log, best_map, stale, start = resume["log"], resume["best_map"], resume["stale"], resume["epoch"] + 1
        rng.bit_generator.state = resume["rng"]
        print(f"resumed after epoch {resume['epoch']} (best val mAP {best_map:.4f})", flush=True)

    head = list(model.classifier.parameters())
    # conv1/conv2 stay frozen (generic low-level filters; halves the backward pass on CPU)
    frozen = [getattr(model, n) for n in ("conv1", "conv2")]
    for p in (p for m in frozen for p in m.parameters()):
        p.requires_grad_(False)
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
    opt = torch.optim.Adam([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}], weight_decay=5e-4)
    iters = args.iters
    total = args.epochs * iters
    step = (start - 1) * iters
    if resume is not None:
        opt.load_state_dict(resume["opt"])
    for epoch in range(start, args.epochs + 1):
        model.train()
        for m in frozen:
            m.eval()
        for p in body:
            p.requires_grad_(epoch > 1)  # classifier-only warm-up epoch: the new head starts random
        order = rng.permutation(len(by_id))
        t0, losses = time.time(), []
        for it in range(iters):
            pids = order[(it * args.p) % len(order) : (it * args.p) % len(order) + args.p]
            if len(pids) < args.p:
                pids = rng.choice(len(by_id), args.p, replace=False)
            idx = np.concatenate([rng.choice(by_id[i], args.k, replace=len(by_id[i]) < args.k) for i in pids])
            x = to_tensor([augment(load_image(tr.path[i]), rng) for i in idx])
            y = torch.as_tensor(tr.label.to_numpy()[idx])
            for g, base in zip(opt.param_groups, (args.lr, args.lr * 10), strict=True):
                g["lr"] = base * 0.5 * (1 + np.cos(np.pi * step / total))
            logits, feat = model(x.to(args.device))
            y = y.to(args.device)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1) + triplet_hard(feat, y, 0.3)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(loss.item())
            if it % 50 == 0:
                print(
                    f"  epoch {epoch} it {it}/{iters} loss {np.mean(losses[-50:]):.3f} {time.time() - t0:.0f}s",
                    flush=True,
                )
        res = validate(model, val)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "sec": time.time() - t0, "val": res})
        print(
            f"epoch {epoch} loss {np.mean(losses):.3f} {time.time() - t0:.0f}s val {json.dumps(res['all'])}", flush=True
        )
        if res["all"]["mAP"] > best_map:
            best_map, stale = res["all"]["mAP"], 0
            torch.save(model.state_dict(), CKPT)
        else:
            stale += 1
        (ROOT / "train_log.json").write_text(json.dumps(log, indent=1))
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "log": log}
        torch.save(state | {"best_map": best_map, "stale": stale, "rng": rng.bit_generator.state}, LAST)
        if stale >= args.patience:
            break
    (ROOT / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"best val mAP {best_map:.4f} → {CKPT}")


def export() -> None:
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = load_factory("osnet_ain_x1_0")(num_classes=state["classifier.weight"].shape[0], pretrained=False)
    model.load_state_dict(state)
    to_onnx(model, paths().models / MODELS["osnet_ain_x1_0_ft"].filename)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "train", "export"])
    ap.add_argument("--per-camera", type=int, default=30, help="max crops per identity per camera")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--iters", type=int, default=150, help="P×K batches per epoch")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.step == "build":
        build(args.per_camera)
    elif args.step == "train":
        train(args)
    else:
        export()


if __name__ == "__main__":
    main()
