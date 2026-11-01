"""Ballbesitz (10 Hz), Aus-Erkennung und Formations-Detektor (Freiwurf, Sprungball)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ..domain.models import Side
from ..domain.observations import Formation, OutOfBounds, PlayerState, Possession
from ..geometry.court import FT_LINE_FROM_BASELINE, HALF_L, LANE_WIDTH, in_court

Pt = tuple[float, float]


@dataclass(frozen=True)
class LastTouch:
    track_id: Optional[int]
    team: Optional[Side]
    player_id: Optional[str]
    video_ts: float


class PossessionEstimator:
    """Besitz = Ball in Reichweite eines Spielers ueber >= ``acquire_frames`` Frames.

    Reichweite: Abstand Ball-Hand-Keypoint (``hand_radius``), wenn Hand-Punkte
    fuer den Track vorliegen, sonst Abstand Ball-Spielerposition (``reach_radius``).
    Ist ein Ball hoeher als ``max_height`` (Wurf/Pass ueber Kopf) oder fehlt er,
    zaehlt der Frame als "frei". Das Loesen braucht ``release_frames`` Frames
    (Hysterese gegen einzelne Fehldetektionen). Emittiert nur bei Aenderung; der
    Zeitstempel ist der des ersten Frames der bestaetigten Serie.
    """

    def __init__(self, reach_radius: float = 1.0, hand_radius: float = 0.5, acquire_frames: int = 3,
                 release_frames: int = 2, max_height: float = 3.2) -> None:
        self.reach_radius, self.hand_radius = reach_radius, hand_radius
        self.acquire_frames, self.release_frames, self.max_height = acquire_frames, release_frames, max_height
        self.owner: Optional[int] = None
        self.last_touch: Optional[LastTouch] = None
        self._cand: Optional[int] = None
        self._count = 0
        self._first_ts = 0.0

    def _nearest(self, players: Sequence[PlayerState], ball: Optional[Pt], hands: Optional[Mapping[int, Sequence[Pt]]],
                 height: Optional[float]):
        if ball is None or (height is not None and height > self.max_height):
            return None, 0.0
        best, bd = None, math.inf
        for p in players:
            if p.team is None:  # Schiris besitzen den Ball nicht
                continue
            hp = hands.get(p.track_id) if hands else None
            if hp:
                d, lim = min(math.hypot(ball[0] - h[0], ball[1] - h[1]) for h in hp), self.hand_radius
            else:
                d, lim = math.hypot(ball[0] - p.x, ball[1] - p.y), self.reach_radius
            if d <= lim and d / lim < bd:
                best, bd = p, d / lim
        return best, (1.0 - bd) if best is not None else 0.0

    def update(self, ts: float, players: Sequence[PlayerState], ball: Optional[Pt],
               hands: Optional[Mapping[int, Sequence[Pt]]] = None,
               ball_height: Optional[float] = None) -> Optional[Possession]:
        p, closeness = self._nearest(players, ball, hands, ball_height)
        c = p.track_id if p else None
        if c == self.owner:
            self._cand, self._count = None, 0
            if p is not None:
                self.last_touch = LastTouch(p.track_id, p.team, p.player_id, ts)
            return None
        if c == self._cand and self._count > 0:
            self._count += 1
        else:
            self._cand, self._count, self._first_ts = c, 1, ts
        need = self.acquire_frames if c is not None else self.release_frames
        if self._count < need:
            return None
        self.owner = c
        self._cand, self._count = None, 0
        if p is None:
            return Possession(self._first_ts, None, None, None, 1.0)
        self.last_touch = LastTouch(p.track_id, p.team, p.player_id, self._first_ts)
        return Possession(self._first_ts, p.track_id, p.team, p.player_id, float(max(0.3, closeness)))


class OutOfBoundsDetector:
    """Ball ausserhalb des Feldpolygons (+``margin``) fuer >= ``min_frames`` Frames
    UND Spielunterbrechung (``play_stopped``, z. B. Uhr steht/Pfiff). Pro Episode
    ein Ereignis; die Unterbrechung darf bis ``stop_window`` s nach Ausgang
    nachlaufen. Zeitpunkt = erster Aus-Frame, letzte Beruehrung = Stand davor."""

    def __init__(self, margin: float = 0.15, min_frames: int = 2, stop_window: float = 3.0) -> None:
        self.margin, self.min_frames, self.stop_window = margin, min_frames, stop_window
        self._out = 0
        self._first_ts = 0.0
        self._touch: Optional[LastTouch] = None
        self._emitted = False

    def update(self, ts: float, ball: Optional[Pt], play_stopped: bool,
               last_touch: Optional[LastTouch] = None) -> Optional[OutOfBounds]:
        if ball is None:
            return None
        outside = not in_court(ball[0], ball[1], self.margin)
        if not outside:
            self._out, self._emitted = 0, False
            return None
        if self._out == 0:
            self._first_ts, self._touch = ts, last_touch
        self._out += 1
        if self._emitted or self._out < self.min_frames or not play_stopped:
            return None
        if ts - self._first_ts > self.stop_window:
            return None
        self._emitted = True
        t = self._touch
        return OutOfBounds(self._first_ts, t.track_id if t else None, t.team if t else None,
                           t.player_id if t else None, 0.8 if t else 0.5)


class FormationDetector:
    """Freiwurf-Aufstellung und Sprungball (Feldkoordinaten).

    Freiwurf: ein Spieler innerhalb ``ft_radius`` (0,6 m) von (+-8,2 | 0) und
    >= ``min_lane`` weitere Spieler an den Zonenraendern (|y| ~ 2,45 +- ``lane_tol``,
    x zwischen Freiwurflinie und Grundlinie). Sprungball: genau zwei Spieler im
    Mittelkreis (r=1,8 m) und alle uebrigen ausserhalb. Die Aufstellung muss
    ``hold`` Sekunden stabil sein; pro Aufstellung ein Ereignis.
    """

    def __init__(self, ft_radius: float = 0.6, min_lane: int = 4, lane_tol: float = 0.6,
                 center_radius: float = 1.8, hold: float = 0.5, min_others: int = 3, gap: float = 1.0) -> None:
        self.ft_radius, self.min_lane, self.lane_tol = ft_radius, min_lane, lane_tol
        self.center_radius, self.hold, self.min_others, self.gap = center_radius, hold, min_others, gap
        self._since: dict[str, float] = {}
        self._last_true: dict[str, float] = {}
        self._fired: set[str] = set()

    def _free_throw(self, players: Sequence[PlayerState]):
        ftx = HALF_L - FT_LINE_FROM_BASELINE
        for basket, sx in (("left", -1), ("right", 1)):
            fp = (sx * ftx, 0.0)
            cand = [p for p in players if math.hypot(p.x - fp[0], p.y - fp[1]) <= self.ft_radius]
            if not cand:
                continue
            shooter = min(cand, key=lambda p: math.hypot(p.x - fp[0], p.y - fp[1]))
            lane_x0, lane_x1 = sorted((sx * ftx, sx * HALF_L))
            lane = [p for p in players if p is not shooter
                    and lane_x0 - 0.3 <= p.x <= lane_x1 + 0.1
                    and abs(abs(p.y) - LANE_WIDTH / 2) <= self.lane_tol]
            if len(lane) >= self.min_lane:
                return basket, shooter, len(lane)
        return None

    def _jump_ball(self, players: Sequence[PlayerState]) -> bool:
        inside = [p for p in players if math.hypot(p.x, p.y) <= self.center_radius]
        outside = len(players) - len(inside)
        if len(inside) != 2 or outside < self.min_others:
            return False
        a, b = inside
        return a.team is None or b.team is None or a.team != b.team

    def update(self, ts: float, players: Sequence[PlayerState]) -> Optional[Formation]:
        ft = self._free_throw(players)
        found = {"free_throw": ft, "jump_ball": self._jump_ball(players) or None}
        result = None
        for kind, hit in found.items():
            if hit:
                self._last_true[kind] = ts
                self._since.setdefault(kind, ts)
                if kind not in self._fired and ts - self._since[kind] >= self.hold:
                    self._fired.add(kind)
                    if kind == "free_throw":
                        basket, sh, n = hit  # type: ignore[misc]
                        result = Formation(self._since[kind], "free_throw", sh.track_id, sh.team, sh.player_id,
                                           basket, min(1.0, 0.5 + 0.1 * n))  # type: ignore[arg-type]
                    else:
                        result = Formation(self._since[kind], "jump_ball", confidence=0.8)
            elif ts - self._last_true.get(kind, -1e9) > self.gap:
                self._since.pop(kind, None)
                self._fired.discard(kind)
        return result
