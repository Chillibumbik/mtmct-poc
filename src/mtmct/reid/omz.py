"""Intel Open Model Zoo `person-reidentification-retail-0287` (256-d); raw BGR 0-255 input per the OMZ docs."""

from __future__ import annotations

from typing import Literal

from mtmct.registry import register
from mtmct.reid.common import OnnxReid


@register("reid", "omz_0287")
class Omz0287(OnnxReid):
    """Intel OMZ person-reidentification-retail-0287 (no PyTorch needed)."""

    LICENCE = "Apache-2.0 (Intel Open Model Zoo)"
    MODEL = "omz_0287"
    DIM = 256

    class Params(OnnxReid.Params):
        backend: Literal["openvino"] = "openvino"  # IR only
