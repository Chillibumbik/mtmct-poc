"""Evaluation (spec §9): per-camera and cross-camera MOT metrics, identity clustering metrics, ReID sanity."""

from mtmct.eval.cross_camera import concat_cameras
from mtmct.eval.identity import identity_metrics
from mtmct.eval.mot import iou_matrix, match_boxes
from mtmct.eval.reid_sanity import reid_sanity
from mtmct.eval.report import evaluate_run, json_safe

__all__ = [
    "concat_cameras",
    "evaluate_run",
    "identity_metrics",
    "iou_matrix",
    "json_safe",
    "match_boxes",
    "reid_sanity",
]
