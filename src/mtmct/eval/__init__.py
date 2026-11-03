"""Evaluation (spec §9): per-camera and cross-camera MOT metrics, identity clustering metrics, ReID sanity."""

from mtmct.eval.identity import identity_metrics
from mtmct.eval.mot import iou_matrix
from mtmct.eval.reid_sanity import reid_sanity
from mtmct.eval.report import evaluate_run, json_safe

__all__ = [
    "evaluate_run",
    "identity_metrics",
    "iou_matrix",
    "json_safe",
    "reid_sanity",
]
