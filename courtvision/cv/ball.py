"""Ball: Parabel-Fit, Wurf-Detektor, Ring-Treffer-Detektor, Heatmap-Spitze.

Der Ball selbst wird von einem trainierten Modell geliefert (RF-DETR-Ballklasse
oder TrackNet-artiges ``BallHeatmapModel``); alles hier arbeitet auf den
resultierenden Punktfolgen.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..domain.models import Side
from ..domain.observations import BallTouch, Possession, PlayerState, ShotRelease, RimMake, Basket, Observation
from ..geometry.court import basket_pos, three_probability

G = 9.81


# -- Parabel -------------------------------------------------------------
@dataclass(frozen=True)
class Parabola:
    """q(t) = a (t-t0)^2 + b (t-t0) + c."""

    a: float
    b: float
    c: float
    t0: float
    rmse: float
    n: int

    def __call__(self, t):
        d = np.asarray(t, float) - self.t0
        return self.a * d * d + self.b * d + self.c

    def slope(self, t: float) -> float:
        return 2 * self.a * (t - self.t0) + self.b

    @property
    def apex_t(self) -> Optional[float]:
        return None if abs(self.a) < 1e-12 else self.t0 - self.b / (2 * self.a)


def fit_parabola(t: Sequence[float], q: Sequence[float], weights: Optional[Sequence[float]] = None) -> Parabola:
    """Kleinste-Quadrate-Parabel q(t). Fuer z(t) (Hoehe) oder, mit t:=x_img,
    q:=y_img, die Bildraum-Bahn y_img(x_img). Mindestens 3 Punkte."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    if len(t) < 3:
        raise ValueError("mindestens 3 Punkte fuer eine Parabel")
    t0 = float(t[0])
    d = t - t0
    A = np.stack([d * d, d, np.ones_like(d)], axis=1)
    w = np.ones_like(d) if weights is None else np.sqrt(np.asarray(weights, float))
    coef, *_ = np.linalg.lstsq(A * w[:, None], q * w, rcond=None)
    rmse = float(np.sqrt(np.mean((A @ coef - q) ** 2)))
    return Parabola(float(coef[0]), float(coef[1]), float(coef[2]), t0, rmse, len(t))


# -- Heatmap (TrackNet-Stil) --------------------------------------------------
@runtime_checkable
class BallHeatmapModel(Protocol):
    def predict(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Drei gestapelte Frames (t-2, t-1, t) -> Heatmap (H, W) mit Werten in [0, 1]."""
        ...


def heatmap_peak(hm: np.ndarray, threshold: float = 0.5) -> Optional[tuple[float, float, float]]:
    """(x, y, peak) der Heatmap-Spitze mit Subpixel-Verfeinerung (Log-Parabel
    durch die drei Stuetzstellen je Achse). None, wenn die Spitze < threshold."""
    hm = np.asarray(hm, float)
    iy, ix = np.unravel_index(int(np.argmax(hm)), hm.shape)
    peak = float(hm[iy, ix])
    if peak < threshold:
        return None
    def refine(m: float, c: float, p: float) -> float:
        # Log-Parabel (exakt fuer Gauss-Spitzen) durch drei Stuetzstellen; Versatz in [-0.5, 0.5]
        if min(m, c, p) <= 0:
            return 0.0
        lm, lc, lp = math.log(m), math.log(c), math.log(p)
        den = lm - 2 * lc + lp
        return 0.0 if den >= 0 else float(np.clip(0.5 * (lm - lp) / den, -0.5, 0.5))

    dx = dy = 0.0
    if 0 < ix < hm.shape[1] - 1:
        dx = refine(hm[iy, ix - 1], peak, hm[iy, ix + 1])
    if 0 < iy < hm.shape[0] - 1:
        dy = refine(hm[iy - 1, ix], peak, hm[iy + 1, ix])
    return float(ix + dx), float(iy + dy), peak


# -- Wurf ---------------------------------------------------------------------
@dataclass
class BallSample:
    ts: float
    x: float  # Feldkoordinaten (Bodenprojektion)
    y: float
    height: Optional[float] = None  # m ueber Boden, falls geschaetzt
    img_y: Optional[float] = None  # Bild-y (nach unten wachsend), Fallback fuer die Vertikale


@dataclass
class _Release:
    ts: float
    track_id: int
    team: Side
    player_id: Optional[str]
    foot: tuple[float, float]
    shot_emitted: bool = False


class ShotDetector:
    """Erkennt Wuerfe aus Ballbahn + Ballbesitz-Wechsel (Besitzer -> frei).

    Ablauf: Endet ein Besitz (``Possession(track_id=None)``), werden die Ballpunkte
    ab Freigabezeit gesammelt. Nach ``min_samples`` Punkten wird die Vertikale als
    Parabel gefittet: Wurf, wenn die Bahn nach oben startet (v_z0 > min_vz),
    konkav ist (a < 0) und die Horizontalbewegung auf einen Korb zeigt
    (cos >= ``min_cos``). Wurfort = Fussposition des Werfers zur Freigabezeit.
    Ohne Hoehenschaetzung wird -img_y als Vertikale benutzt (``min_vz_img`` in px/s).
    Zusaetzlich: abrupter Bahnabbruch nahe dem Werfer -> BallTouch (Blockhinweis).
    """

    def __init__(self, min_samples: int = 4, max_window: float = 0.8, min_vz: float = 1.5,
                 min_vz_img: float = 60.0, min_vx_img: float = 1.5, min_cos: float = 0.6, sigma: float = 0.25,
                 block_window: float = 0.8, block_dist: float = 2.5, abrupt_dv: float = 3.0,
                 foot_history: float = 1.5) -> None:
        self.min_samples, self.max_window = min_samples, max_window
        self.min_vz, self.min_vz_img, self.min_cos, self.sigma = min_vz, min_vz_img, min_cos, sigma
        self.min_vx_img = min_vx_img
        self.block_window, self.block_dist, self.abrupt_dv = block_window, block_dist, abrupt_dv
        self.foot_history = foot_history
        self._balls: deque[BallSample] = deque(maxlen=400)
        self._holder: Optional[tuple[int, Side, Optional[str]]] = None
        self._feet: deque[tuple[float, tuple[float, float]]] = deque(maxlen=200)
        self._rel: Optional[_Release] = None

    # -- Analyse einer Ballfolge -----------------------------------------
    def _vertical(self, s: Sequence[BallSample]):
        if all(b.height is not None for b in s):
            return np.array([b.height for b in s], float), self.min_vz
        if all(b.img_y is not None for b in s):
            return -np.array([b.img_y for b in s], float), self.min_vz_img
        return None, 0.0

    def analyze(self, samples: Sequence[BallSample]) -> Optional[tuple[Basket, float]]:
        """Ist die Folge eine Wurfbahn? -> (Korb, Konfidenz) oder None."""
        if len(samples) < 3:
            return None
        z, min_vz = self._vertical(samples)
        if z is None:
            return None
        t = np.array([b.ts for b in samples], float)
        fit = fit_parabola(t, z)
        vz0 = fit.slope(t[0])
        if vz0 < min_vz or fit.a >= 0:  # muss aufwaerts starten und konkav sein
            return None
        xy = np.array([[b.x, b.y] for b in samples])
        A = np.stack([t - t[0], np.ones_like(t)], axis=1)
        vx = np.linalg.lstsq(A, xy[:, 0], rcond=None)[0][0]
        vy = np.linalg.lstsq(A, xy[:, 1], rcond=None)[0][0]
        if samples[0].height is None:
            # Nur Bild-Vertikale: die Bodenprojektion eines fliegenden Balls ist in y
            # durch die Hoehe verfaelscht -> nur die x-Komponente (Grundlinie zu
            # Grundlinie) und die Richtung zum Korb werden benutzt.
            if abs(vx) < self.min_vx_img:
                return None
            basket = "right" if vx > 0 else "left"
            bx, _ = basket_pos(basket)
            if (bx - xy[0, 0]) * vx <= 0:
                return None
            conf = float(np.clip(0.3 + 0.4 * min(1.0, abs(vx) / 6.0) + 0.2 * min(1.0, vz0 / (3 * min_vz)), 0.05, 0.8))
            return basket, conf  # type: ignore[return-value]
        speed = math.hypot(vx, vy)
        if speed < 0.5:
            return None
        best, best_cos = None, -1.0
        for basket in ("left", "right"):
            bx, by = basket_pos(basket)
            dx, dy = bx - xy[0, 0], by - xy[0, 1]
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            c = (vx * dx + vy * dy) / (speed * d)
            if c > best_cos:
                best, best_cos = basket, c
        if best is None or best_cos < self.min_cos:
            return None
        conf = float(np.clip(0.5 * best_cos + 0.5 * min(1.0, vz0 / (3 * min_vz)), 0.05, 1.0))
        return best, conf  # type: ignore[return-value]

    def _abrupt_index(self, s: Sequence[BallSample]) -> Optional[int]:
        """Index des ersten Samples, an dem die Horizontalbewegung abrupt abbricht."""
        for k in range(2, len(s)):
            dt1, dt2 = s[k - 1].ts - s[k - 2].ts, s[k].ts - s[k - 1].ts
            if dt1 <= 0 or dt2 <= 0:
                continue
            v1 = np.array([s[k - 1].x - s[k - 2].x, s[k - 1].y - s[k - 2].y]) / dt1
            v2 = np.array([s[k].x - s[k - 1].x, s[k].y - s[k - 1].y]) / dt2
            if np.linalg.norm(v2 - v1) > max(self.abrupt_dv, 0.7 * np.linalg.norm(v1)) and np.linalg.norm(v1) > 1.0:
                return k
        return None

    # -- Streaming ------------------------------------------------------------
    def update(self, ts: float, ball: Optional[BallSample], players: Sequence[PlayerState] = (),
               possessions: Sequence[Possession] = ()) -> list[Observation]:
        out: list[Observation] = []
        if ball is not None:
            self._balls.append(ball)
        if self._holder is not None:
            for p in players:
                if p.track_id == self._holder[0]:
                    self._feet.append((ts, (p.x, p.y)))
        for ev in possessions:
            if ev.track_id is not None:
                self._holder = (ev.track_id, ev.team, ev.player_id)  # type: ignore[assignment]
                self._rel = None
            else:  # Besitz endet
                if self._holder is not None and self._holder[1] is not None:
                    foot = self._foot_at(ev.video_ts)
                    if foot is not None:
                        self._rel = _Release(ev.video_ts, self._holder[0], self._holder[1], self._holder[2], foot)
                self._holder = None
        r = self._rel
        if r is None:
            return out
        samples = [b for b in self._balls if b.ts >= r.ts - 1e-6]
        if not r.shot_emitted:
            k = self._abrupt_index(samples)
            usable = samples[:k] if k is not None else samples
            ready = k is not None or len(samples) >= self.min_samples or ts - r.ts >= self.max_window
            if ready:
                res = self.analyze(usable[: max(self.min_samples, 3) + 4 if k is None else len(usable)])
                if res is not None:
                    basket, conf = res
                    out.append(ShotRelease(r.ts, r.track_id, r.team, r.foot[0], r.foot[1], basket,
                                           r.player_id, three_probability(r.foot[0], r.foot[1], basket, self.sigma),
                                           conf))
                    r.shot_emitted = True
                elif k is None and (len(samples) >= self.min_samples or ts - r.ts >= self.max_window):
                    self._rel = None  # kein Wurf (Pass/Dribbling)
                    return out
                elif k is not None:
                    self._rel = None
                    return out
        if r.shot_emitted:
            k = self._abrupt_index(samples)
            if k is not None and samples[k].ts - r.ts <= self.block_window:
                b = samples[k]
                if math.hypot(b.x - r.foot[0], b.y - r.foot[1]) <= self.block_dist:
                    d = self._nearest_defender(b, r.team, players)
                    if d is not None:
                        out.append(BallTouch(b.ts, d.track_id, d.team, d.player_id, True, 0.6))  # type: ignore[arg-type]
                self._rel = None
            elif ts - r.ts > self.block_window:
                self._rel = None
        return out

    def _foot_at(self, ts: float) -> Optional[tuple[float, float]]:
        cand = [f for f in self._feet if f[0] <= ts + 1e-6]
        if cand:
            return cand[-1][1]
        return self._feet[0][1] if self._feet else None

    @staticmethod
    def _nearest_defender(b: BallSample, shooter_team: Side, players: Sequence[PlayerState]) -> Optional[PlayerState]:
        opp = [p for p in players if p.team is not None and p.team != shooter_team]
        if not opp:
            return None
        d = min(opp, key=lambda p: math.hypot(p.x - b.x, p.y - b.y))
        return d if math.hypot(d.x - b.x, d.y - b.y) <= 2.0 else None


# -- Ring-Treffer ---------------------------------------------------------------
class RimMakeDetector:
    """Ballbahn in der Ring-ROI (Bildkoordinaten, y nach unten).

    Treffer-Kandidat: Ball war oberhalb der Ringebene, kreuzt sie abwaertsgerichtet
    innerhalb des Ringradius (``inside_frac``) und bleibt danach darunter bzw. wird
    vom Netz verdeckt (``lost_confirm`` Sekunden ohne Detektion). Prallt der Ball
    nach oben ab oder kreuzt von unten nach oben, gibt es keinen Treffer.
    """

    def __init__(self, basket: Basket, rim_center: tuple[float, float], rim_radius: float,
                 inside_frac: float = 0.9, depth_frac: float = 0.5, lost_confirm: float = 0.15,
                 pending_timeout: float = 0.6, min_above: float = 0.3) -> None:
        self.basket, self.c, self.r = basket, rim_center, rim_radius
        self.inside_frac, self.depth_frac, self.lost_confirm = inside_frac, depth_frac, lost_confirm
        self.pending_timeout, self.min_above = pending_timeout, min_above
        self._prev: Optional[tuple[float, float, float]] = None
        self._max_above = 0.0
        self._pending: Optional[tuple[float, float]] = None  # (crossing ts, offset)
        self._last_seen = -1e9
        self._cooldown_until = -1e9

    def update(self, ts: float, uv: Optional[tuple[float, float]]) -> Optional[RimMake]:
        cy = self.c[1]
        if uv is None:
            if self._pending and ts - self._last_seen >= self.lost_confirm:
                return self._fire(ts)
            if self._pending and ts - self._pending[0] > self.pending_timeout:
                self._pending = None
            return None
        u, v = uv
        self._last_seen = ts
        above = (cy - v) / self.r  # >0: ueber der Ringebene
        if self._pending:
            if above < -self.depth_frac:
                return self._fire(ts)
            if above > 0.1 or ts - self._pending[0] > self.pending_timeout:
                self._pending = None  # Ball geht wieder hoch: Abpraller
        elif self._prev is not None and ts >= self._cooldown_until:
            pt, pu, pv = self._prev
            pabove = (cy - pv) / self.r
            if pabove > 0 and above <= 0 and self._max_above >= self.min_above:
                f = pabove / (pabove - above)
                uc = pu + f * (u - pu)
                off = (uc - self.c[0]) / self.r
                if abs(off) <= self.inside_frac:
                    self._pending = (pt + f * (ts - pt), off)
                    if above < -self.depth_frac:
                        return self._fire(ts)
        if above > 0:
            self._max_above = max(self._max_above, above)
        elif above < -1.0 and self._pending is None:
            self._max_above = 0.0
        self._prev = (ts, u, v)
        return None

    def _fire(self, ts: float) -> RimMake:
        t, off = self._pending  # type: ignore[misc]
        self._pending = None
        self._cooldown_until = ts + 1.0
        self._max_above = 0.0
        return RimMake(t, self.basket, float(np.clip(1.0 - 0.5 * abs(off), 0.3, 1.0)))
