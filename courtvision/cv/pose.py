"""Pose: Protocol fuer RTMPose-artige Keypoint-Modelle plus COCO-17-Helfer."""
from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KP = {n: i for i, n in enumerate(COCO_KEYPOINTS)}


@runtime_checkable
class PoseEstimator(Protocol):
    def estimate(self, frame: np.ndarray, boxes: Sequence[tuple[float, float, float, float]]) -> list[np.ndarray]:
        """Je Box ein Array (17, 3) mit (x, y, conf) in Bildpixeln."""
        ...


class RTMPoseEstimator:  # pragma: no cover - benoetigt Modell
    """Optionaler Adapter auf ``rtmlib`` (RTMPose, Apache-2.0). Best effort:
    die rtmlib-API variiert je Version; Import erst bei Konstruktion."""

    def __init__(self, onnx_model: str, input_size: tuple[int, int] = (192, 256),
                 backend: str = "onnxruntime", device: str = "cpu") -> None:
        try:
            from rtmlib import RTMPose  # type: ignore
        except ImportError as e:
            raise ImportError("RTMPoseEstimator benoetigt 'rtmlib' (pip install rtmlib onnxruntime).") from e
        self._pose = RTMPose(onnx_model, model_input_size=input_size, backend=backend, device=device)

    def estimate(self, frame, boxes):
        kps, scores = self._pose(frame, bboxes=list(boxes))
        return [np.concatenate([k, s[:, None]], axis=1) for k, s in zip(kps, scores)]


def _valid(kp: np.ndarray, name: str, min_conf: float) -> bool:
    return bool(kp[KP[name], 2] >= min_conf)


def hand_points(kp: np.ndarray, min_conf: float = 0.3, extend: float = 0.25) -> list[tuple[float, float]]:
    """Handpunkte: Handgelenk, um ``extend`` x Unterarm verlaengert (COCO hat
    keine Hand-Keypoints). Nur Gelenke mit ausreichender Konfidenz."""
    pts = []
    for side in ("left", "right"):
        w, e = f"{side}_wrist", f"{side}_elbow"
        if not _valid(kp, w, min_conf):
            continue
        p = kp[KP[w], :2].astype(float)
        if _valid(kp, e, min_conf):
            p = p + extend * (p - kp[KP[e], :2])
        pts.append((float(p[0]), float(p[1])))
    return pts


def foot_point(kp: np.ndarray, min_conf: float = 0.3) -> Optional[tuple[float, float]]:
    """Bodenpunkt: Mittel der Knoechel (sonst einer); None ohne gueltigen Knoechel."""
    pts = [kp[KP[n], :2] for n in ("left_ankle", "right_ankle") if _valid(kp, n, min_conf)]
    if not pts:
        return None
    m = np.mean(pts, axis=0)
    return float(m[0]), float(m[1])
