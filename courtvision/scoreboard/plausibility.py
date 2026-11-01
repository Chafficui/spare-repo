"""Plausibilitätsregeln mit Entprellung für Anzeigetafel-Messwerte.

Regeln: Punkte steigen nur um 1-3; die Uhr läuft monoton ab oder steht (Reset
nur beim Viertelwechsel); das Viertel steigt nur um 1; Teamfouls steigen um 1
und werden je Viertel zurückgesetzt. Ein Wert wird erst nach ``n_stable``
identischen Messungen übernommen. Unübliche Änderungen (+4 Punkte, Punktabzug,
Uhr rückwärts, Viertelsprung) brauchen ``n_unusual`` Messungen und werden dann
mit einem :class:`Anomaly` gemeldet (die Tafel kann korrekt von einem Bediener
korrigiert worden sein); einzelne Ausreisser werden verworfen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .state import ScoreboardReading

_ORDER = ("period", "home", "away", "fouls_home", "fouls_away", "clock")


@dataclass
class Anomaly:
    kind: str  # z. B. score_jump, score_decrease, clock_increase, period_jump
    field: str
    video_ts: float
    observed: object
    expected: object
    accepted: bool  # False: verworfen/noch nicht übernommen, True: nach Entprellung übernommen
    message: str = ""


@dataclass
class FilterOutput:
    reading: Optional[ScoreboardReading]  # akzeptierter Gesamtzustand (None: noch nichts)
    changed: set[str] = field(default_factory=set)
    anomalies: list[Anomaly] = field(default_factory=list)


class PlausibilityFilter:
    def __init__(self, n_stable: int = 3, n_unusual: Optional[int] = None, min_conf: float = 0.5,
                 clock_tolerance: float = 2.0, max_clock_speed: float = 1.0):
        self.n_stable = n_stable
        self.n_unusual = n_unusual if n_unusual is not None else 2 * n_stable
        self.min_conf = min_conf
        self.clock_tolerance = clock_tolerance
        self.max_clock_speed = max_clock_speed
        self.acc: dict[str, float] = {}
        self._cand: dict[str, tuple[float, int, float]] = {}
        self._clock_obs_ts: Optional[float] = None
        self._clock_reset_ok = False
        self._foul_reset_ok = False

    def update(self, r: ScoreboardReading) -> FilterOutput:
        out = FilterOutput(None)
        for name in _ORDER:
            v = getattr(r, name)
            if v is None or r.confidence(name) < self.min_conf:
                continue
            self._step(name, v, r, out)
        if self.acc:
            out.reading = ScoreboardReading(
                video_ts=r.video_ts, running=r.running, source=r.source,
                conf={k: r.conf.get(k, 1.0) for k in self.acc},
                **{k: (v if k == "clock" else int(v)) for k, v in self.acc.items()})
        return out

    # ------------------------------------------------------------------
    def _judge(self, name: str, v: float, cur: float, r: ScoreboardReading) -> tuple[str, str]:
        """-> (Urteil, Anomalie-Art); Urteil "ok" oder "unusual"."""
        if name in ("home", "away"):
            d = v - cur
            if 1 <= d <= 3:
                return "ok", ""
            return "unusual", "score_jump" if d > 3 else "score_decrease"
        if name == "period":
            if v == cur + 1:
                return "ok", ""
            return "unusual", "period_jump" if v > cur else "period_decrease"
        if name in ("fouls_home", "fouls_away"):
            if v == cur + 1:
                return "ok", ""
            if v < cur and v == 0 and self._foul_reset_ok:
                return "ok", ""
            return "unusual", "foul_jump" if v > cur else "foul_decrease"
        # clock
        if self._clock_reset_ok:
            return "ok", ""
        if v > cur:
            return "unusual", "clock_increase"
        last = self._clock_obs_ts if self._clock_obs_ts is not None else r.video_ts
        allowed = max(0.0, r.video_ts - last) * self.max_clock_speed + self.clock_tolerance
        if cur - v > allowed:
            return "unusual", "clock_jump"
        return "ok", ""

    def _same(self, name: str, v: float, cand: tuple, r: ScoreboardReading) -> bool:
        if name != "clock":
            return v == cand[0]
        drop = cand[0] - v  # Kandidat gilt weiter, wenn die Uhr plausibel weiterläuft
        return -1e-9 <= drop <= max(0.0, r.video_ts - cand[2]) * self.max_clock_speed + self.clock_tolerance

    def _step(self, name: str, v: float, r: ScoreboardReading, out: FilterOutput) -> None:
        cur = self.acc.get(name)
        if cur is not None and v == cur:
            self._cand.pop(name, None)
            if name == "clock":
                self._clock_obs_ts = r.video_ts
            return
        if cur is None:
            verdict, kind = "ok", ""
        else:
            verdict, kind = self._judge(name, v, cur, r)
        c = self._cand.get(name)
        count = c[1] + 1 if c and self._same(name, v, c, r) else 1
        self._cand[name] = (v, count, r.video_ts)
        # Die laufende Uhr ändert sich jedes Frame: Entprellung nur für unübliche Sprünge.
        need = (1 if name == "clock" else self.n_stable) if verdict == "ok" else self.n_unusual
        if verdict == "unusual" and count == 1:
            out.anomalies.append(Anomaly(kind, name, r.video_ts, v, cur, False,
                                         f"{name}: {cur} -> {v} unplausibel, warte auf Bestätigung"))
        if count < need:
            return
        # akzeptieren
        self._cand.pop(name, None)
        self.acc[name] = v
        out.changed.add(name)
        if verdict == "unusual":
            out.anomalies.append(Anomaly(kind, name, r.video_ts, v, cur, True,
                                         f"{name}: {cur} -> {v} nach Entprellung übernommen"))
        if name == "period" and cur is not None:
            self._clock_reset_ok = True
            self._foul_reset_ok = True
        elif name == "clock":
            self._clock_reset_ok = False
            self._clock_obs_ts = r.video_ts
        elif name in ("fouls_home", "fouls_away"):
            self._foul_reset_ok = False
