"""Fine-tune YOLOX-s (official COCO weights, Apache-2.0) as a person detector on GT of the tune splits; export ONNX.

    uv run --group export python scripts/finetune_yolox.py build                  # frames + YOLO labels → data/_ft/det/
    uv run --group export python scripts/finetune_yolox.py train --variant general
    uv run --group export python scripts/finetune_yolox.py train --variant wildtrack --init general
    uv run --group export python scripts/finetune_yolox.py export --variant general  # → models/yolox_s_ft.onnx

Why this route: the model/loss code is YOLOX's own (Apache-2.0, fetched at a pinned commit and checked by sha256,
like `export_osnet.py` does for torchreid), and the exported ONNX has the official layout (80 classes, raw head
output), so the existing `detector/yolox_s` pre/post-processing serves it unchanged. Only GT of
`mtmct.data.finetune.TUNE_SOURCES` is used. WILDTRACK annotates only a 12×36 m zone with projected-cylinder boxes,
so its frames would teach "unannotated person = background": the `general` variant leaves WILDTRACK out, the
`wildtrack` variant continues from `general` on WILDTRACK tune frames (plus replay of the general set).
Recipe: SGD (momentum 0.9, nesterov), cosine LR with warm-up, BN frozen (batch 8), random scale/translate, flip,
HSV jitter; no mosaic. Every 10th sampled frame is held out for validation (recall/precision at IoU 0.5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from mtmct.config import paths
from mtmct.data.base import load_sequence
from mtmct.data.finetune import TUNE_SOURCES, read_frames, tune_gt, write_yolo_labels
from mtmct.detect.yolox import postprocess
from mtmct.models import MODELS, _download

ROOT = paths().data / "_ft" / "det"
SIZE = 640
YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
YOLOX_FILES = {
    "models/darknet.py": "368c9db45166e863691cf7460b314d047928f90e8744c492a673b6eff9b809f3",
    "models/network_blocks.py": "f490240a674799f1b548ac4ca571c7f3e59d4e75efee8ebeb1e24e6e4e10d2a4",
    "models/yolo_pafpn.py": "8571af557f6caacda61e54813be4be2d73c4a0889e1f876ec5586057eb2eebde",
    "models/yolo_head.py": "0275bcdeb1b1d5dbd3172fc4897c75cb161f707843983bf582fb087676f969c2",
    "models/yolox.py": "9f8eaf76d2c243627e875db9b522cf1cfddc0cdcf229400550f76102cce741e3",
    "models/losses.py": "7fa685fb4653bab69d5880c3f749edeb8a4cabfe71fadf0acc5151c2eaa75702",
    "utils/boxes.py": "564fc1605b14ebe7414c42f7098adb70647e27f3d796df8dc4611795e78e3a86",
    "utils/compat.py": "a8f0dec9e0566cf0a09a211bc08ce47bc7abd6dd75d239ee85d8c0c5fe3e07ce",
}
INIT_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
INIT_SHA = "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"
# images per tune source (≤ 1500 in total); WILDTRACK feeds only the `wildtrack` variant
BUDGET = {"campus": 300, "aic24": 400, "chirla": 250, "epfl_terrace": 250, "wildtrack": 300}
VARIANTS = {"general": "yolox_s_ft", "wildtrack": "yolox_s_ft_wt"}


def yolox_modules() -> types.SimpleNamespace:
    """Import YOLOX model code from pinned upstream files; loguru/torchvision/visualize are not needed for training."""
    root = paths().models / "_src" / f"yolox_{YOLOX_COMMIT[:8]}"
    for rel, sha in YOLOX_FILES.items():
        path = root / "yolox" / rel
        if not path.exists():
            url = f"https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/{YOLOX_COMMIT}/yolox/{rel}"
            _download(url, path, sha)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"{path} does not match the pinned YOLOX source")
    (root / "yolox" / "__init__.py").write_text("")
    (root / "yolox" / "models" / "__init__.py").write_text("")
    (root / "yolox" / "utils" / "__init__.py").write_text(
        "from .boxes import bboxes_iou, cxcywh2xyxy\nfrom .compat import meshgrid\n\n\n"
        "def visualize_assign(*args, **kwargs):\n    raise NotImplementedError\n"
    )
    sys.modules.setdefault("loguru", types.SimpleNamespace(logger=types.SimpleNamespace(error=print, info=print)))  # type: ignore[arg-type]
    sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
    sys.path.insert(0, str(root))
    from yolox.models.yolo_head import YOLOXHead
    from yolox.models.yolo_pafpn import YOLOPAFPN
    from yolox.models.yolox import YOLOX

    return types.SimpleNamespace(YOLOX=YOLOX, YOLOPAFPN=YOLOPAFPN, YOLOXHead=YOLOXHead)


def build_model(init: Path | None) -> torch.nn.Module:
    m = yolox_modules()
    model = m.YOLOX(m.YOLOPAFPN(0.33, 0.50), m.YOLOXHead(80, 0.50))
    if init is None:
        init = paths().models / "_src" / "yolox_s.pth"
        if not init.exists():
            _download(INIT_URL, init, INIT_SHA)
    state = torch.load(init, map_location="cpu", weights_only=False)
    # denormal floats make CPU convs several times slower (seen in the OSNet checkpoint)
    model.load_state_dict({k: torch.where(v.abs() < 1e-30, 0, v) for k, v in state.get("model", state).items()})
    return model


def build() -> None:
    rows = []
    for (ds, name), split in TUNE_SOURCES.items():
        seq = load_sequence(paths().data, ds, name)
        gt = tune_gt(seq, split)
        t0, t1 = seq.split_range(split)
        n_cam = BUDGET[ds] // len(seq.cameras) + 1
        (ROOT / ds / "images").mkdir(parents=True, exist_ok=True)
        (ROOT / ds / "labels").mkdir(parents=True, exist_ok=True)
        tic = time.time()
        for cam in seq.cameras:
            fids = np.flatnonzero(
                (cam.ts(np.arange(cam.n_frames)) >= t0) & (cam.ts(np.arange(cam.n_frames)) < t1 - 1e-6)
            )
            pick = fids[np.linspace(0, len(fids) - 1, n_cam).round().astype(int)]
            boxes = gt[gt.camera_id == cam.camera_id]
            for i, (fid, image) in enumerate(read_frames(seq, cam, pick)):
                stem = f"{cam.camera_id}_{fid:06d}"
                scale = SIZE / max(image.shape[:2])
                small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(ROOT / ds / "images" / f"{stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
                b = boxes[boxes.frame_id == fid]
                write_yolo_labels(ROOT / ds / "labels" / f"{stem}.txt", b, cam.width, cam.height)
                rows.append({"dataset": ds, "stem": stem, "camera_id": cam.camera_id, "frame_id": int(fid),
                             "n_boxes": len(b), "val": i % 10 == 5})  # fmt: skip
        print(f"{ds}/{name} split={split}: {sum(r['dataset'] == ds for r in rows)} images, {time.time() - tic:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "index.csv", index=False)
    print(df.groupby(["dataset", "val"]).agg(images=("stem", "size"), boxes=("n_boxes", "sum")))


def load_sample(ds: str, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Image (BGR, longer side 640) and boxes (n, 4) xyxy in its pixels."""
    image = cv2.imread(str(ROOT / ds / "images" / f"{stem}.jpg"))
    text = (ROOT / ds / "labels" / f"{stem}.txt").read_text()
    lab = np.array(text.split(), dtype=np.float64).reshape(-1, 5)
    h, w = image.shape[:2]
    cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
    return image, np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def augment(image: np.ndarray, boxes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random scale (0.6-1.5) + placement on a 640 canvas (pad 114), flip, HSV jitter → (canvas, xyxy boxes)."""
    h, w = image.shape[:2]
    s = SIZE / max(h, w) * rng.uniform(0.6, 1.5)
    nw, nh = round(w * s), round(h * s)
    img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (
        rng.integers(min(0, SIZE - nw), max(0, SIZE - nw) + 1),
        rng.integers(min(0, SIZE - nh), max(0, SIZE - nh) + 1),
    )
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    x0, y0 = max(dx, 0), max(dy, 0)
    x1, y1 = min(dx + nw, SIZE), min(dy + nh, SIZE)
    canvas[y0:y1, x0:x1] = img[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    b = boxes * s + [dx, dy, dx, dy]
    clipped = np.clip(b, [x0, y0, x0, y0], [x1, y1, x1, y1])
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep_area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    b = clipped[(keep_area > 0.35 * area) & (clipped[:, 2] - clipped[:, 0] > 3) & (clipped[:, 3] - clipped[:, 1] > 6)]
    if rng.random() < 0.5:
        canvas = canvas[:, ::-1]
        b = np.stack([SIZE - b[:, 2], b[:, 1], SIZE - b[:, 0], b[:, 3]], 1)
    hsv = cv2.cvtColor(np.ascontiguousarray(canvas), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
    hsv[..., 1:] *= rng.uniform(0.7, 1.3, 2)
    canvas = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return canvas, b


def letterbox_top_left(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    r = SIZE / max(h, w)
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    canvas[: round(h * r), : round(w * r)] = cv2.resize(image, (round(w * r), round(h * r)))
    return canvas, r


def targets(boxes: list[np.ndarray], max_labels: int = 150) -> torch.Tensor:
    t = torch.zeros(len(boxes), max_labels, 5)
    for i, b in enumerate(boxes):
        b = b[:max_labels]
        t[i, : len(b), 1] = torch.from_numpy((b[:, 0] + b[:, 2]) / 2)
        t[i, : len(b), 2] = torch.from_numpy((b[:, 1] + b[:, 3]) / 2)
        t[i, : len(b), 3] = torch.from_numpy(b[:, 2] - b[:, 0])
        t[i, : len(b), 4] = torch.from_numpy(b[:, 3] - b[:, 1])
    return t  # class 0 = person


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


@torch.no_grad()
def validate(model: torch.nn.Module, val: pd.DataFrame, conf: float = 0.3) -> dict:
    """Recall / precision at IoU 0.5 per dataset, with the inference post-processing of `detector/yolox_s`."""
    model.eval()
    model.head.decode_in_inference = False
    stats: dict[str, list[int]] = {}
    for r in val.itertuples():
        image, gt = load_sample(r.dataset, r.stem)
        canvas, ratio = letterbox_top_left(image)
        x = torch.from_numpy(canvas.transpose(2, 0, 1).astype(np.float32))[None]
        raw = model(x.to(next(model.parameters()).device))[0].cpu().numpy()
        det = postprocess(raw, SIZE, ratio, image.shape[:2], conf, 0.45)
        pred = np.concatenate([det[:, :2], det[:, :2] + det[:, 2:4]], 1)
        tp = 0
        if len(pred) and len(gt):  # greedy matching by IoU
            iou = iou_matrix(gt, pred)
            while iou.size and iou.max() >= 0.5:
                i, j = np.unravel_index(iou.argmax(), iou.shape)
                tp += 1
                iou[i, :], iou[:, j] = 0, 0
        s = stats.setdefault(r.dataset, [0, 0, 0])
        s[0] += tp
        s[1] += len(gt)
        s[2] += len(pred)
    out = {ds: {"recall": tp / max(n, 1), "precision": tp / max(p, 1)} for ds, (tp, n, p) in stats.items()}
    tot = np.sum(list(stats.values()), axis=0)
    out["all"] = {"recall": tot[0] / max(tot[1], 1), "precision": tot[0] / max(tot[2], 1)}
    out["all"]["f1"] = (
        2 * out["all"]["recall"] * out["all"]["precision"] / max(out["all"]["recall"] + out["all"]["precision"], 1e-9)
    )
    return out


def freeze_bn(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def ckpt(variant: str) -> Path:
    return paths().models / "_src" / f"{VARIANTS[variant]}.pth"


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    df = pd.read_csv(ROOT / "index.csv")
    general = df.dataset != "wildtrack"
    if args.variant == "general":
        tr, val = df[general & ~df.val], df[general & df.val]
    else:  # WILDTRACK tune frames + an equal-size replay of the general set
        wt = df[~general & ~df.val]
        tr = pd.concat([wt, df[general & ~df.val].sample(len(wt), random_state=0)])
        val = df[df.val]
    tr, val = tr.reset_index(drop=True), val.reset_index(drop=True)
    model = build_model(ckpt(args.init) if args.init else None)
    last = ckpt(args.variant).with_suffix(".last.pth")  # resume state after every epoch
    resume = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    if resume is None:
        model.to(args.device)
        log = [{"epoch": 0, "val": validate(model, val)}]
        print(f"epoch 0 (init) val {json.dumps(log[0]['val'])}", flush=True)
        best, start = log[0]["val"]["all"]["f1"], 1
        torch.save({"model": model.state_dict()}, ckpt(args.variant))
    else:
        model.load_state_dict(resume["model"])
        log, best, start = resume["log"], resume["best"], resume["epoch"] + 1
        rng.bit_generator.state = resume["rng"]
        print(f"resumed after epoch {resume['epoch']} (best val F1 {best:.4f})", flush=True)
    model.to(args.device)
    if args.freeze_backbone:
        for p in model.backbone.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0}],
        lr=args.lr, momentum=0.9, nesterov=True,
    )  # fmt: skip
    iters = len(tr) // args.batch
    total, warm, step = args.epochs * iters, min(100, iters), (start - 1) * iters
    if resume is not None:
        opt.load_state_dict(resume["opt"])
    deadline = time.time() + args.hours * 3600
    for epoch in range(start, args.epochs + 1):
        model.train()
        model.head.decode_in_inference = True
        freeze_bn(model)
        order = rng.permutation(len(tr))
        tic, losses = time.time(), []
        for it in range(iters):
            batch = [
                augment(*load_sample(tr.dataset[i], tr.stem[i]), rng)
                for i in order[it * args.batch : (it + 1) * args.batch]
            ]
            x = torch.from_numpy(np.stack([b[0] for b in batch]).transpose(0, 3, 1, 2).astype(np.float32))
            lr = args.lr * (
                step / warm if step < warm else 0.5 * (1 + np.cos(np.pi * (step - warm) / max(total - warm, 1)))
            )
            for g in opt.param_groups:
                g["lr"] = lr
            out = model(x.to(args.device), targets([b[1] for b in batch]).to(args.device))
            opt.zero_grad()
            out["total_loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
            if not torch.isfinite(norm):  # a rare degenerate assignment gives inf/NaN gradients: skip that batch
                print(f"  epoch {epoch} it {it}: non-finite gradient, batch skipped", flush=True)
                continue
            opt.step()
            step += 1
            losses.append(float(out["total_loss"]))
            if it % 20 == 0:
                print(
                    f"  epoch {epoch} it {it}/{iters} loss {np.mean(losses[-20:]):.3f} lr {lr:.5f} {time.time() - tic:.0f}s",  # noqa: E501
                    flush=True,
                )
        res = validate(model, val)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "sec": time.time() - tic, "val": res})
        print(f"epoch {epoch} loss {np.mean(losses):.3f} {time.time() - tic:.0f}s val {json.dumps(res)}", flush=True)
        if res["all"]["f1"] > best:
            best = res["all"]["f1"]
            torch.save({"model": model.state_dict()}, ckpt(args.variant))
        (ROOT / f"train_log_{args.variant}.json").write_text(json.dumps(log, indent=1))
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "log": log, "best": best}
        torch.save(state | {"rng": rng.bit_generator.state}, last)
        if time.time() + log[-1]["sec"] + 60 > deadline:
            print("time budget reached", flush=True)
            break
    print(f"best val F1 {best:.4f} → {ckpt(args.variant)}")


def export(variant: str) -> None:
    model = build_model(ckpt(variant))
    model.eval()
    model.head.decode_in_inference = False  # raw head output, like the official ONNX
    out = paths().models / MODELS[VARIANTS[variant]].filename
    dummy = torch.zeros(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model, (dummy,), str(out), input_names=["images"], output_names=["output"], opset_version=17, dynamo=False
    )
    import onnxruntime as ort

    x = torch.rand(1, 3, SIZE, SIZE) * 255
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
    err = float(np.abs(ref - got).max())
    if got.shape != (1, 8400, 85) or err > 1e-2:
        out.unlink()
        raise RuntimeError(f"{out.name}: ONNX output differs from PyTorch (shape {got.shape}, max abs err {err:.2e})")
    print(f"{out}: sha256={hashlib.sha256(out.read_bytes()).hexdigest()} max_abs_err={err:.1e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "train", "export"])
    ap.add_argument("--variant", choices=list(VARIANTS), default="general")
    ap.add_argument("--init", choices=list(VARIANTS), default=None, help="start from a fine-tuned variant")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hours", type=float, default=3.0, help="per launch: stop after the epoch that would cross it")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.step == "build":
        build()
    elif args.step == "train":
        train(args)
    else:
        export(args.variant)


if __name__ == "__main__":
    main()
