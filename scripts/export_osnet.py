"""Export torchreid OSNet MSMT17 checkpoints to ONNX (opset 17, dynamic batch). Run once per machine:

    uv run --group export python scripts/export_osnet.py [osnet_ain_x1_0 osnet_x0_25]

The network definition is torchreid's (MIT, (c) 2018 Kaiyang Zhou, github.com/KaiyangZhou/deep-person-reid). Instead of
installing torchreid (Cython build) the two source files are fetched at a pinned commit and checked by sha256.
Weights come from the authors' HF repo `kaiyangzhou/osnet` (see `mtmct.models.MODELS`). Output: `models/<filename>`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import httpx
import numpy as np
import torch

from mtmct.config import paths
from mtmct.models import MODELS, fetch_source

COMMIT = "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"
# model name -> (torchreid module, factory, sha256 of the module file at COMMIT)
ARCH = {
    "osnet_ain_x1_0": ("osnet_ain", "a43b495866a4b1645cd134a30c3011592e8c6826ba4781955a135588cf46cd25"),
    "osnet_x0_25": ("osnet", "c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae"),
}


def load_factory(name: str):
    module, sha = ARCH[name]
    path = paths().models / "_src" / f"torchreid_{COMMIT[:8]}_{module}.py"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/{COMMIT}/torchreid/models/{module}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(httpx.get(url, follow_redirects=True, timeout=60).raise_for_status().content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError(f"{path} does not match the pinned torchreid source")
    spec = importlib.util.spec_from_file_location(f"torchreid_{module}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def export(name: str) -> Path:
    state = torch.load(fetch_source(name), map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = load_factory(name)(num_classes=state["classifier.weight"].shape[0], pretrained=False)
    model.load_state_dict(state)
    model.eval()

    out = paths().models / MODELS[name].filename
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(2, 3, 256, 128)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["images"],
        output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(5, 3, 256, 128)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"images": x.numpy()})[0]
    err = float(np.abs(ref - got).max())
    if got.shape != (5, 512) or err > 1e-3:
        out.unlink()
        raise RuntimeError(f"{name}: ONNX output differs from PyTorch (shape {got.shape}, max abs err {err:.2e})")
    print(f"{name}: {out} sha256={hashlib.sha256(out.read_bytes()).hexdigest()} max_abs_err={err:.1e}")
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or list(ARCH):
        export(n)
