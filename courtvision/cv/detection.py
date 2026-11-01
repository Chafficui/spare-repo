"""Detektion: Datentypen, Detector-Protocol, RF-DETR-Adapter, Aufloesungsstrategie.

Aufloesungsstrategie (Produktplan): Die Detektion laeuft auf dem herunter-
skalierten Vollbild. Fuer Trikotnummern und Ball werden gezielt Ausschnitte in
voller Aufloesung gelesen (``crop_regions``); die Ringe bekommen permanente
ROIs mit voller Bildrate. Alles hier ist reines numpy-Slicing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, NamedTuple, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

CLASSES = ("player", "referee", "ball", "rim")
BBox = tuple[float, float, float, float]


@dataclass
class Detection:
    """Eine Detektion; ``bbox`` = (x1, y1, x2, y2) in Bildpixeln."""

    bbox: BBox
    score: float = 1.0
    cls: str = "player"

    def __post_init__(self) -> None:
        if self.cls not in CLASSES:
            raise ValueError(f"unbekannte Klasse {self.cls!r}; erlaubt: {CLASSES}")
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        self.bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Unterkante Mitte: Bodenkontakt eines Spielers."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    def scaled(self, sx: float, sy: Optional[float] = None) -> "Detection":
        sy = sx if sy is None else sy
        x1, y1, x2, y2 = self.bbox
        return Detection((x1 * sx, y1 * sy, x2 * sx, y2 * sy), self.score, self.cls)


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detektionen eines RGB-Frames (H, W, 3) in dessen Pixelkoordinaten."""
        ...


class DetectorUnavailableError(ImportError):
    pass


class RFDETRDetector:
    """Adapter fuer RF-DETR (Apache-2.0, https://github.com/roboflow/rf-detr).

    ``rfdetr`` wird erst hier importiert; ohne Installation gibt es eine klare
    Fehlermeldung. ``class_names`` muss zum feinjustierten Checkpoint passen
    (Index = Klassen-ID); Namen ausserhalb von ``CLASSES`` werden verworfen.
    """

    SIZES = ("nano", "small", "medium", "base", "large")
    _CLASS_ATTR = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
                   "base": "RFDETRBase", "large": "RFDETRLarge"}

    def __init__(self, size: str = "base", checkpoint_path: Optional[str] = None,
                 threshold: float = 0.4, class_names: Sequence[str] = ("ball", "player", "referee", "rim"),
                 lazy: bool = False) -> None:
        if size not in self.SIZES:
            raise ValueError(f"size muss eine von {self.SIZES} sein")
        self.size, self.checkpoint_path = size, checkpoint_path
        self.threshold, self.class_names = threshold, tuple(class_names)
        self._model = None
        if not lazy:
            self.load()

    def load(self):
        if self._model is not None:
            return self._model
        try:
            import rfdetr  # type: ignore
        except ImportError as e:  # pragma: no cover - depends on environment
            raise DetectorUnavailableError(
                "RFDETRDetector benoetigt das optionale Paket 'rfdetr' (pip install rfdetr). "
                "Fuer Tests/Offline-Betrieb stattdessen einen eigenen Detector oder "
                "CVPipeline.process_tracks verwenden.") from e
        cls = getattr(rfdetr, self._CLASS_ATTR[self.size])  # pragma: no cover
        kwargs = {"pretrain_weights": self.checkpoint_path} if self.checkpoint_path else {}  # pragma: no cover
        self._model = cls(**kwargs)  # pragma: no cover
        return self._model  # pragma: no cover

    def detect(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - needs model
        model = self.load()
        res = model.predict(frame, threshold=self.threshold)
        out: list[Detection] = []
        for box, score, cid in zip(res.xyxy, res.confidence, res.class_id):
            cid = int(cid)
            if 0 <= cid < len(self.class_names) and self.class_names[cid] in CLASSES:
                out.append(Detection(tuple(float(v) for v in box), float(score), self.class_names[cid]))
        return out


class Downscaled(NamedTuple):
    image: np.ndarray
    scale_x: float  # full_width / small_width (Faktor Klein -> Voll)
    scale_y: float


def downscale(frame: np.ndarray, max_side: Optional[int] = None, factor: Optional[float] = None) -> Downscaled:
    """Verkleinert ein Bild (H, W[, C]). Ganzzahlige Faktoren: Box-Mittelung,
    sonst Nearest-Neighbour. Rueckgabe enthaelt die Faktoren zum Zurueckrechnen."""
    h, w = frame.shape[:2]
    if factor is None:
        factor = max(h, w) / max_side if max_side else 1.0
    if factor <= 1.0:
        return Downscaled(frame, 1.0, 1.0)
    if abs(factor - round(factor)) < 1e-9:
        f = int(round(factor))
        hh, ww = (h // f) * f, (w // f) * f
        v = frame[:hh, :ww].astype(np.float32)
        v = v.reshape(hh // f, f, ww // f, f, *frame.shape[2:]).mean(axis=(1, 3))
        small = v.astype(frame.dtype) if np.issubdtype(frame.dtype, np.integer) else v
        return Downscaled(small, w / (ww // f), h / (hh // f))
    nh, nw = max(1, int(round(h / factor))), max(1, int(round(w / factor)))
    ys = np.minimum((np.arange(nh) * h / nh).astype(int), h - 1)
    xs = np.minimum((np.arange(nw) * w / nw).astype(int), w - 1)
    return Downscaled(frame[ys][:, xs], w / nw, h / nh)


@dataclass
class CropRegion:
    kind: str  # "jersey" | "ball" | "rim"
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in Volbild-Pixeln
    image: np.ndarray = field(repr=False)
    track_id: Optional[int] = None
    name: Optional[str] = None  # z. B. "left"/"right" bei Ring-ROIs


def _clip_box(b, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1, x2 = int(max(0, np.floor(x1))), int(min(w, np.ceil(x2)))
    y1, y2 = int(max(0, np.floor(y1))), int(min(h, np.ceil(y2)))
    return x1, y1, max(x1, x2), max(y1, y2)


def rim_roi(center_uv: tuple[float, float], radius_px: float, scale: float = 3.0) -> BBox:
    """Quadratische ROI (Volbild) um einen Ring."""
    r = radius_px * scale
    return (center_uv[0] - r, center_uv[1] - r, center_uv[0] + r, center_uv[1] + r)


def crop_regions(frame: np.ndarray, *, scale: tuple[float, float] | float = 1.0,
                 player_boxes: Optional[Mapping[int, BBox]] = None,
                 ball_candidates: Sequence[tuple[float, float]] = (),
                 rim_rois: Optional[Mapping[str, BBox]] = None,
                 jersey_band: tuple[float, float] = (0.12, 0.60), pad: float = 0.1,
                 ball_size: int = 96) -> list[CropRegion]:
    """Volbild-Ausschnitte gemaess Aufloesungsstrategie.

    ``player_boxes`` und ``ball_candidates`` stehen in Koordinaten des
    herunterskalierten Bildes (mit ``scale`` = Faktor Klein->Voll); ``rim_rois``
    sind bereits Volbild-Koordinaten und werden *immer* geliefert (permanent,
    volle Bildrate). Jersey-Crop = oberes Koerperband (Ruecken-/Brustnummer).
    """
    sx, sy = (scale, scale) if np.isscalar(scale) else scale
    h, w = frame.shape[:2]
    out: list[CropRegion] = []
    for tid, (x1, y1, x2, y2) in (player_boxes or {}).items():
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
        bw, bh = x2 - x1, y2 - y1
        box = _clip_box((x1 - pad * bw, y1 + jersey_band[0] * bh, x2 + pad * bw, y1 + jersey_band[1] * bh), w, h)
        if box[2] > box[0] and box[3] > box[1]:
            out.append(CropRegion("jersey", box, frame[box[1]:box[3], box[0]:box[2]], track_id=tid))
    half = ball_size // 2
    for cx, cy in ball_candidates:
        cx, cy = cx * sx, cy * sy
        box = _clip_box((cx - half, cy - half, cx + half, cy + half), w, h)
        if box[2] > box[0] and box[3] > box[1]:
            out.append(CropRegion("ball", box, frame[box[1]:box[3], box[0]:box[2]]))
    for name, b in (rim_rois or {}).items():
        box = _clip_box(b, w, h)
        out.append(CropRegion("rim", box, frame[box[1]:box[3], box[0]:box[2]], name=name))
    return out
