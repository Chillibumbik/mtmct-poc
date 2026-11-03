"""OSNet body-ReID (torchreid, MSMT17 weights) exported to ONNX by `scripts/export_osnet.py`; 512-d embeddings."""

from __future__ import annotations

from mtmct.registry import register
from mtmct.reid.common import OnnxReid

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _Osnet(OnnxReid):
    LICENCE = "code MIT (torchreid); weights trained on MSMT17, research use only"
    DIM = 512
    MEAN = IMAGENET_MEAN
    STD = IMAGENET_STD


@register("reid", "osnet_ain_x1_0")
class OsnetAinX10(_Osnet):
    """OSNet-AIN x1.0, MSMT17 (best cross-domain generalisation of the family)."""

    MODEL = "osnet_ain_x1_0"


@register("reid", "osnet_x0_25")
class OsnetX025(_Osnet):
    """OSNet x0.25, MSMT17 (fastest)."""

    MODEL = "osnet_x0_25"


@register("reid", "osnet_ain_x1_0_ft")
class OsnetAinX10Ft(_Osnet):
    """OSNet-AIN x1.0 fine-tuned on GT crops of the tune splits (`scripts/finetune_reid.py`); same preprocessing."""

    LICENCE = "code MIT (torchreid); MSMT17 init + fine-tuned on open datasets (tune splits), research use only"
    MODEL = "osnet_ain_x1_0_ft"
