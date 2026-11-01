"""Virtuelle Kamera: geglaettetes Crop-Fenster aus dem 4K-Weitwinkelbild (reines numpy)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

Point = Sequence[float]


@dataclass(frozen=True)
class CropRect:
    x: float
    y: float
    w: float
    h: float

    def ints(self) -> tuple[int, int, int, int]:
        return int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h))


def _even(v: float) -> int:
    return int(round(v / 2.0)) * 2


@dataclass
class CameraConfig:
    frame_w: int = 3840
    frame_h: int = 2160
    fps: float = 25.0
    crop_w: int = 1920
    crop_h: int = 1080
    #: Hochformat 9:16 (folgt dem Hauptakteur)
    vertical_h: int = 1620
    max_pan_speed: float = 1500.0  # Pixel/s (Quellbild)
    smooth_time: float = 0.6  # s, Zeitkonstante der kritisch gedaempften Feder
    ball_weight: float = 3.0
    player_weight: float = 1.0
    nearest_players: int = 6  # nur die n Spieler, die dem Ball am naechsten sind

    def size(self, vertical: bool) -> tuple[int, int]:
        if vertical:
            return _even(self.vertical_h * 9 / 16), self.vertical_h
        return self.crop_w, self.crop_h


class _SmoothDamp:
    """Kritisch gedaempfte Feder (SmoothDamp) mit hartem Geschwindigkeitslimit."""

    def __init__(self, pos: np.ndarray, smooth_time: float, max_speed: float) -> None:
        self.pos = np.asarray(pos, float)
        self.vel = np.zeros(2)
        self.st = max(1e-3, smooth_time)
        self.max_speed = max_speed

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        omega = 2.0 / self.st
        x = omega * dt
        ex = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x ** 3)
        change = self.pos - target
        max_change = self.max_speed * self.st
        norm = float(np.hypot(*change))
        if norm > max_change:
            change = change * (max_change / norm)
        tgt = self.pos - change
        temp = (self.vel + omega * change) * dt
        self.vel = (self.vel - omega * temp) * ex
        out = tgt + (change + temp) * ex
        # Overshoot verhindern
        if np.dot(target - self.pos, out - target) > 0:
            out = target.copy()
            self.vel = np.zeros(2)
        # hartes Limit pro Frame
        step = out - self.pos
        sn = float(np.hypot(*step))
        lim = self.max_speed * dt
        if sn > lim:
            out = self.pos + step * (lim / sn)
            self.vel = (out - self.pos) / dt
        self.pos = out
        return out


def _target(ball: Optional[Point], players: Sequence[Point], actor: Optional[Point],
            cfg: CameraConfig, mode: str) -> Optional[np.ndarray]:
    pl = np.asarray(players, float).reshape(-1, 2) if len(players) else np.zeros((0, 2))
    if mode == "actor":
        if actor is not None:
            return np.asarray(actor, float)
        return pl.mean(axis=0) if len(pl) else None
    pts, ws = [], []
    if ball is not None:
        b = np.asarray(ball, float)
        pts.append(b)
        ws.append(cfg.ball_weight)
        if len(pl) > cfg.nearest_players:
            pl = pl[np.argsort(np.hypot(*(pl - b).T))[: cfg.nearest_players]]
    for p in pl:
        pts.append(p)
        ws.append(cfg.player_weight)
    if not pts:
        return None
    return np.average(np.asarray(pts), axis=0, weights=np.asarray(ws))


def virtual_camera(
    ball: Sequence[Optional[Point]],
    players: Sequence[Sequence[Point]],
    cfg: Optional[CameraConfig] = None,
    mode: str = "ball",
    actor: Optional[Sequence[Optional[Point]]] = None,
) -> list[CropRect]:
    """Crop-Rechtecke je Frame (Koordinaten im Quellbild).

    ``ball[i]``: Bildposition (px) oder None; ``players[i]``: Bildpositionen aller
    Spieler; ``mode="actor"`` (Hochformat 9:16) folgt ``actor[i]`` statt dem Ball.
    Ziel = gewichteter Schwerpunkt, geglaettet per kritisch gedaempfter Feder mit
    maximaler Schwenkgeschwindigkeit, am Bildrand geklemmt.
    """
    cfg = cfg or CameraConfig()
    n = max(len(ball), len(players), len(actor) if actor is not None else 0)
    w, h = cfg.size(vertical=(mode == "actor"))
    lo = np.array([w / 2.0, h / 2.0])
    hi = np.array([cfg.frame_w - w / 2.0, cfg.frame_h - h / 2.0])
    dt = 1.0 / cfg.fps
    rects: list[CropRect] = []
    damp: Optional[_SmoothDamp] = None
    last = np.array([cfg.frame_w / 2.0, cfg.frame_h / 2.0])
    for i in range(n):
        tgt = _target(ball[i] if i < len(ball) else None,
                      players[i] if i < len(players) else [],
                      actor[i] if actor is not None and i < len(actor) else None, cfg, mode)
        if tgt is not None:
            last = np.clip(tgt, lo, hi)
        if damp is None:
            damp = _SmoothDamp(last.copy(), cfg.smooth_time, cfg.max_pan_speed)
            c = damp.pos
        else:
            c = np.clip(damp.step(last, dt), lo, hi)
            damp.pos = c
        rects.append(CropRect(float(c[0] - w / 2.0), float(c[1] - h / 2.0), float(w), float(h)))
    return rects
