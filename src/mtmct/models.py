"""Model weights manifest, download with sha256 check, and a thin OpenVINO / ONNX Runtime session wrapper.

`kind="onnx"`: `url` is the final model file (ONNX, or an OpenVINO IR `.xml` with its `.bin` in `companions`).
`kind="export"`: `url`/`sha256` describe the source PyTorch checkpoint, and `filename` is the ONNX produced by
`scripts/export_osnet.py` (needs the `export` dependency group).
`kind="trained"`: fine-tuned locally on the tune splits of open datasets; `url` is the command that rebuilds the file
and `sha256` identifies the file the reported numbers came from (retraining gives other weights, so it is not checked).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import numpy as np

from mtmct.config import paths

_YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
_OMZ = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0287/FP32"
_OSNET = "https://huggingface.co/kaiyangzhou/osnet/resolve/main"
_OSNET_LICENCE = "code MIT (torchreid); weights trained on MSMT17 (research use only)"
EXPORT_HINT = "uv run --group export python scripts/export_osnet.py"
_FT_LICENCE = "fine-tuned on open datasets (tune splits), research use"
_FT_REID = "uv run --group export python scripts/finetune_reid.py {build,train,export}"
_FT_DET = "uv run --group export python scripts/finetune_yolox.py {build,train,export}"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    licence: str
    filename: str
    kind: Literal["onnx", "export", "trained"] = "onnx"
    companions: tuple[tuple[str, str], ...] = ()  # extra (url, sha256) saved next to the main file, same stem


MODELS: dict[str, ModelSpec] = {
    s.name: s
    for s in (
        ModelSpec(
            "yolox_s",
            f"{_YOLOX}/yolox_s.onnx",
            "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
            "Apache-2.0",
            "yolox_s.onnx",
        ),
        ModelSpec(
            "yolox_tiny",
            f"{_YOLOX}/yolox_tiny.onnx",
            "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
            "Apache-2.0",
            "yolox_tiny.onnx",
        ),
        # The OMZ ONNX of this model gives NaN in ONNX Runtime and differs from the IR: use the IR (OpenVINO only).
        ModelSpec(
            "omz_0287",
            f"{_OMZ}/person-reidentification-retail-0287.xml",
            "09af746c1068a3e89355e1f0226d3d6982dc1cd77634b1f70a40f1f654a94be1",
            "Apache-2.0 (Intel Open Model Zoo)",
            "person-reidentification-retail-0287.xml",
            companions=(
                (
                    f"{_OMZ}/person-reidentification-retail-0287.bin",
                    "7f53d18d11253455e0e11646b2176cf6509cc7c97cb5376f9b91b4c41e09a06c",
                ),
            ),
        ),
        ModelSpec(
            "osnet_ain_x1_0",
            f"{_OSNET}/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
            "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
            _OSNET_LICENCE,
            "osnet_ain_x1_0_msmt17.onnx",
            "export",
        ),
        ModelSpec(
            "osnet_x0_25",
            f"{_OSNET}/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
            "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18",
            _OSNET_LICENCE,
            "osnet_x0_25_msmt17.onnx",
            "export",
        ),
        ModelSpec(
            "osnet_ain_x1_0_ft",
            _FT_REID,
            "daf50607027575e16c2f1476f3debfcfe8af9aa16fbab73b881f0a3955d73a8b",
            f"{_OSNET_LICENCE}; {_FT_LICENCE}",
            "osnet_ain_x1_0_ft.onnx",
            "trained",
        ),
        ModelSpec(
            "yolox_s_ft",
            _FT_DET,
            "7d8aef682fa056b1d98aaf86ec80b144385e3e2b3bcd8a8e30b9699a668d40c0",
            f"Apache-2.0 (YOLOX, COCO init); {_FT_LICENCE}",
            "yolox_s_ft.onnx",
            "trained",
        ),
        ModelSpec(
            "yolox_s_ft_wt",
            _FT_DET,
            "b1e6459ffa00efce52681795e5d54e205c6c7e68180074b64605ced47de4a331",
            f"Apache-2.0 (YOLOX, COCO init); {_FT_LICENCE}; WILDTRACK annotation convention",
            "yolox_s_ft_wt.onnx",
            "trained",
        ),
    )
}


def _spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}") from None


def _download(url: str, dest: Path, sha256: str) -> Path:
    """Stream `url` to `dest` (atomic rename after the sha256 matches)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r, part.open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_bytes(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    if digest.hexdigest() != sha256:
        part.unlink()
        raise ValueError(f"sha256 mismatch for {url}: got {digest.hexdigest()}, expected {sha256}")
    part.rename(dest)
    return dest


def fetch_source(name: str) -> Path:
    """Download the source checkpoint of an `export`-kind model into `models/_src/` and return its path."""
    spec = _spec(name)
    dest = paths().models / "_src" / spec.url.rsplit("/", 1)[-1]
    return dest if dest.exists() else _download(spec.url, dest, spec.sha256)


def ensure_model(name: str) -> Path:
    """Path of the ready-to-run model file; downloads `onnx`-kind models on first use."""
    spec = _spec(name)
    path = paths().models / spec.filename
    if path.exists():
        return path
    if spec.kind == "export":
        raise FileNotFoundError(f"{path} is missing: it is exported from PyTorch weights, run `{EXPORT_HINT}`")
    if spec.kind == "trained":
        raise FileNotFoundError(f"{path} is missing: it is produced by fine-tuning, run `{spec.url}`")
    for url, sha256 in spec.companions:
        _download(url, path.with_suffix(Path(url).suffix), sha256)
    return _download(spec.url, path, spec.sha256)


class Runner:
    """One ONNX model on OpenVINO or ONNX Runtime; `__call__` takes a float32 NCHW batch, returns output 0.

    `input_shape` reshapes the model (OpenVINO only; ORT must already match). A static batch dimension is
    served by running the rows one by one.
    """

    def __init__(
        self,
        path: Path,
        backend: Literal["openvino", "onnxruntime"] = "openvino",
        device: str = "CPU",
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        self._static_batch = False
        if backend == "openvino":
            import openvino as ov

            core = ov.Core()
            model = core.read_model(str(path))
            if input_shape is not None:
                model.reshape(list(input_shape))
            self._static_batch = model.input().partial_shape[0].is_static
            compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
            self._run = lambda x: compiled(x)[0]
        elif backend == "onnxruntime":
            import onnxruntime as ort

            if device != "CPU":
                raise ValueError("onnxruntime backend supports device='CPU' only")
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            if input_shape is not None and any(
                isinstance(d, int) and d != want and want > 0 for d, want in zip(inp.shape, input_shape, strict=True)
            ):
                raise ValueError(
                    f"onnxruntime cannot reshape {path.name}: model input {inp.shape}, wanted {input_shape}"
                )
            self._static_batch = isinstance(inp.shape[0], int)
            self._run = lambda x: sess.run(None, {inp.name: x})[0]
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._static_batch and len(x) != 1:
            return np.concatenate([self._run(x[i : i + 1]) for i in range(len(x))])
        return np.asarray(self._run(x))


def load_runner(
    name: str,
    backend: Literal["openvino", "onnxruntime"] = "openvino",
    device: str = "CPU",
    input_shape: tuple[int, ...] | None = None,
) -> Runner:
    return Runner(ensure_model(name), backend, device, input_shape)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download all default model weights into models/")
    ap.add_argument("names", nargs="*", help=f"subset of {sorted(MODELS)} (default: all)")
    names = ap.parse_args(argv).names or list(MODELS)
    for name in names:
        try:
            print(f"{name}: {ensure_model(name)}")
        except FileNotFoundError as e:  # export-kind weights need the optional torch group; do not fail setup
            print(f"{name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
