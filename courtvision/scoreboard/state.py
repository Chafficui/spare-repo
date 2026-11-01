"""Anzeigetafel-Zustand: ein (optischer oder serieller) Messwert der Tafel."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from courtvision.domain.events import Event, EventSource, EventStatus, EventType
from courtvision.domain.observations import ClockState

#: Felder mit Wert (ohne ``running``, das nicht per OCR gelesen wird).
VALUE_FIELDS = ("home", "away", "clock", "period", "fouls_home", "fouls_away")


@dataclass
class ScoreboardReading:
    """Eine Messung der Tafel. Jedes Feld ist optional (None = nicht lesbar).

    ``conf`` hält die Konfidenz je Feld (0..1); fehlt ein Eintrag für ein
    vorhandenes Feld, gilt 1.0 (z. B. Datenschnittstelle).
    ``clock`` ist die Restzeit in Sekunden (Bruchteile bei "SS.t").
    """

    video_ts: float
    home: Optional[int] = None
    away: Optional[int] = None
    clock: Optional[float] = None
    period: Optional[int] = None
    running: Optional[bool] = None
    fouls_home: Optional[int] = None
    fouls_away: Optional[int] = None
    conf: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"

    def confidence(self, name: str) -> float:
        if getattr(self, name) is None:
            return 0.0
        return float(self.conf.get(name, 1.0))

    @property
    def overall_confidence(self) -> float:
        vals = [self.confidence(n) for n in VALUE_FIELDS if getattr(self, n) is not None]
        return min(vals) if vals else 0.0

    def replace(self, **changes) -> "ScoreboardReading":
        return dataclasses.replace(self, **changes)

    def to_event(self, game_id: str) -> Event:
        """SCOREBOARD-Event (source SCOREBOARD, status CONFIRMED)."""
        return Event(
            game_id=game_id,
            type=EventType.SCOREBOARD,
            video_ts=self.video_ts,
            period=int(self.period or 0),
            clock=self.clock,
            payload={
                "home": self.home,
                "away": self.away,
                "clock": self.clock,
                "period": self.period,
                "running": bool(self.running) if self.running is not None else False,
                "fouls_home": self.fouls_home,
                "fouls_away": self.fouls_away,
            },
            confidence=self.overall_confidence,
            source=EventSource.SCOREBOARD,
            status=EventStatus.CONFIRMED,
        )

    def to_clock_state(self) -> Optional[ClockState]:
        """ClockState, sofern Viertel und Uhr bekannt sind, sonst None."""
        if self.period is None or self.clock is None:
            return None
        return ClockState(
            video_ts=self.video_ts,
            period=int(self.period),
            clock=float(self.clock),
            running=bool(self.running),
        )
