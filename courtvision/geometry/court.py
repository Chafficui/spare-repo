"""FIBA-Spielfeld, Dreierlinie und Kalibrierung (Homographie).

Koordinatensystem: Meter, Ursprung Feldmitte, x entlang der Länge
(-14..14), y entlang der Breite (-7.5..7.5). Linker Korb bei x<0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

COURT_LENGTH = 28.0
COURT_WIDTH = 15.0
HALF_L = COURT_LENGTH / 2
HALF_W = COURT_WIDTH / 2
BASKET_FROM_BASELINE = 1.575
THREE_RADIUS = 6.75
THREE_CORNER_FROM_SIDELINE = 0.90
THREE_CORNER_Y = HALF_W - THREE_CORNER_FROM_SIDELINE  # 6.6
FT_LINE_FROM_BASELINE = 5.80
LANE_WIDTH = 4.90
RESTRICTED_RADIUS = 1.25

BASKET_X = HALF_L - BASKET_FROM_BASELINE  # 12.425


def basket_pos(basket: str) -> tuple[float, float]:
    return (-BASKET_X, 0.0) if basket == "left" else (BASKET_X, 0.0)


def nearest_basket(x: float) -> str:
    return "left" if x < 0 else "right"


def _three_signed_distance(x: float, y: float, basket: str) -> float:
    """Vorzeichenbehafteter Abstand zur Dreierlinie (>0: Dreierbereich).

    Linienpolygon: Bogen r=6.75 um den Korb, in den Ecken gerade Stücke
    parallel zur Seitenlinie bei |y|=6.6 bis zur Grundlinie.
    """
    bx, _ = basket_pos(basket)
    dx = abs(x - bx)
    # Punkte hinter der Korbebene (zur Grundlinie hin) zählen über Ecklinie
    ay = abs(y)
    # Schnittpunkt Bogen/Ecklinie: dx_c = sqrt(r^2 - 6.6^2) ~ 1.415 Richtung Feldmitte
    dx_c = math.sqrt(THREE_RADIUS**2 - THREE_CORNER_Y**2)
    toward_center = (x - bx) * (-1 if bx > 0 else 1)  # >0 Richtung Feldmitte
    if toward_center <= dx_c:
        # Eckbereich: Gerade bei |y| = 6.6
        return ay - THREE_CORNER_Y
    return math.hypot(dx, y) - THREE_RADIUS


def is_three(x: float, y: float, basket: str) -> bool:
    """True, wenn der Wurfort hinter der Dreierlinie liegt (Linie selbst = 2P)."""
    return _three_signed_distance(x, y, basket) > 0.0


def three_probability(x: float, y: float, basket: str, sigma: float = 0.25) -> float:
    """P(Dreier) bei Positionsunsicherheit sigma (m) – Unsicherheitsband."""
    d = _three_signed_distance(x, y, basket)
    if sigma <= 0:
        return 1.0 if d > 0 else 0.0
    return 0.5 * (1.0 + math.erf(d / (sigma * math.sqrt(2.0))))


def in_court(x: float, y: float, margin: float = 0.0) -> bool:
    return abs(x) <= HALF_L + margin and abs(y) <= HALF_W + margin


def in_lane(x: float, y: float, basket: str) -> bool:
    bx = HALF_L if basket == "right" else -HALF_L
    ftx = bx - FT_LINE_FROM_BASELINE if basket == "right" else bx + FT_LINE_FROM_BASELINE
    lo, hi = sorted((ftx, bx))
    return lo <= x <= hi and abs(y) <= LANE_WIDTH / 2


def distance_to_basket(x: float, y: float, basket: str) -> float:
    bx, by = basket_pos(basket)
    return math.hypot(x - bx, y - by)


def point_in_polygon(x: float, y: float, poly: Sequence[tuple[float, float]]) -> bool:
    """Ray casting; für Bank-/Wechselzonen aus der Kalibrierung."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xin = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xin:
                inside = not inside
        j = i
    return inside


def three_point_line(basket: str = "right", n: int = 64) -> list[tuple[float, float]]:
    """Polylinie der Dreierlinie (für Shot Charts / Overlays)."""
    bx, _ = basket_pos(basket)
    s = 1 if bx > 0 else -1
    dx_c = math.sqrt(THREE_RADIUS**2 - THREE_CORNER_Y**2)
    pts = [(s * HALF_L, -THREE_CORNER_Y), (bx - s * dx_c, -THREE_CORNER_Y)]
    a0 = math.asin(THREE_CORNER_Y / THREE_RADIUS)
    for i in range(n + 1):
        a = -a0 + 2 * a0 * i / n
        pts.append((bx - s * THREE_RADIUS * math.cos(a), THREE_RADIUS * math.sin(a)))
    pts += [(bx - s * dx_c, THREE_CORNER_Y), (s * HALF_L, THREE_CORNER_Y)]
    return pts


#: Benannte Feldpunkte für die Klick-Kalibrierung im Setup.
LANDMARKS: dict[str, tuple[float, float]] = {
    "corner_top_left": (-HALF_L, HALF_W),
    "corner_top_right": (HALF_L, HALF_W),
    "corner_bottom_left": (-HALF_L, -HALF_W),
    "corner_bottom_right": (HALF_L, -HALF_W),
    "center": (0.0, 0.0),
    "midline_top": (0.0, HALF_W),
    "midline_bottom": (0.0, -HALF_W),
    "ft_left_top": (-HALF_L + FT_LINE_FROM_BASELINE, LANE_WIDTH / 2),
    "ft_left_bottom": (-HALF_L + FT_LINE_FROM_BASELINE, -LANE_WIDTH / 2),
    "ft_right_top": (HALF_L - FT_LINE_FROM_BASELINE, LANE_WIDTH / 2),
    "ft_right_bottom": (HALF_L - FT_LINE_FROM_BASELINE, -LANE_WIDTH / 2),
    "lane_left_top_baseline": (-HALF_L, LANE_WIDTH / 2),
    "lane_left_bottom_baseline": (-HALF_L, -LANE_WIDTH / 2),
    "lane_right_top_baseline": (HALF_L, LANE_WIDTH / 2),
    "lane_right_bottom_baseline": (HALF_L, -LANE_WIDTH / 2),
}


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = math.sqrt(2) / d if d > 0 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[pts, np.ones(len(pts))] @ T.T
    return ph, T


def fit_homography(src: Sequence[tuple[float, float]], dst: Sequence[tuple[float, float]]) -> np.ndarray:
    """Normalisierte DLT, mindestens 4 Punktpaare. Liefert H mit dst ~ H @ src."""
    src_a = np.asarray(src, dtype=float)
    dst_a = np.asarray(dst, dtype=float)
    if len(src_a) < 4 or len(src_a) != len(dst_a):
        raise ValueError("mindestens 4 Punktpaare gleicher Anzahl nötig")
    s, Ts = _normalize(src_a)
    d, Td = _normalize(dst_a)
    rows = []
    for (x, y, _), (u, v, _) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: Iterable[tuple[float, float]]) -> np.ndarray:
    p = np.asarray(list(pts), dtype=float)
    ph = np.c_[p, np.ones(len(p))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


@dataclass
class Calibration:
    """Feste Kamerakalibrierung: Bild (Pixel) <-> Feld (Meter).

    ``zones`` enthält Polygone in Feldkoordinaten: "substitution",
    "bench_home", "bench_away" (aus dem Setup markiert).
    """

    image_points: list[tuple[float, float]]
    court_points: list[tuple[float, float]]
    zones: Optional[dict[str, list[tuple[float, float]]]] = None

    def __post_init__(self) -> None:
        self.H = fit_homography(self.image_points, self.court_points)
        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def from_landmarks(cls, clicks: dict[str, tuple[float, float]], zones=None) -> "Calibration":
        names = [n for n in clicks if n in LANDMARKS]
        return cls([clicks[n] for n in names], [LANDMARKS[n] for n in names], zones)

    def to_court(self, u: float, v: float) -> tuple[float, float]:
        x, y = apply_homography(self.H, [(u, v)])[0]
        return float(x), float(y)

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        u, v = apply_homography(self.H_inv, [(x, y)])[0]
        return float(u), float(v)

    def reprojection_error(self, image_points=None, court_points=None) -> float:
        """Mittlerer Fehler in Pixeln – Driftkontrolle gegen Keypoint-Modell."""
        ip = np.asarray(image_points or self.image_points, float)
        cp = court_points or self.court_points
        proj = apply_homography(self.H_inv, cp)
        return float(np.sqrt(((proj - ip) ** 2).sum(axis=1)).mean())

    def zone_of(self, x: float, y: float) -> Optional[str]:
        for name, poly in (self.zones or {}).items():
            if point_in_polygon(x, y, poly):
                return name
        return None
