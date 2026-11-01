"""Spiel-/Periodengrenzen aus dem Tafel- bzw. Uhrstrom ableiten.

Bisher emittiert niemand ``GAME_START``/``PERIOD_START``/``PERIOD_END``/``GAME_END``;
ohne sie rechnet der Box-Score Minuten nur bis zum letzten Event der Schlussperiode.
Der :class:`PeriodTracker` leitet sie aus ``ClockState``/``ScoreboardReading`` ab
(source SCOREBOARD, status CONFIRMED, ``period``/``clock`` gesetzt, Uhr 0.0 bei Enden).

Regeln
------
* ``GAME_START``: erste laufende Uhr in Periode 1 (zusammen mit ``PERIOD_START``).
* ``PERIOD_START``: erste laufende Uhr einer Periode.
* ``PERIOD_END``: Uhr 0 und stehend, oder die Periodennummer steigt (dann wird die
  Endzeit aus der zuletzt laufenden Uhr geschaetzt: ``ts_letzte + Restzeit``).
  Nur fuer Perioden, die gestartet wurden.
* ``GAME_END``: nach dem Ende der letzten Regelperiode (oder einer Verlaengerung)
  bei nicht ausgeglichenem Stand. Die Tafel zeigt den letzten Korb oft erst
  verzoegert, daher wird erst ``end_settle_s`` Videosekunden nach dem Periodenende
  entschieden (Stand muss bekannt sein); ``end()`` erzwingt es. Das Event traegt
  den ``video_ts`` des Periodenendes. Danach ignoriert der Tracker alles.
* Idempotent: je (Typ, Periode) hoechstens ein Event; wiederholte Messwerte erzeugen nichts.
"""
from __future__ import annotations

from typing import Optional, Union

from courtvision.domain.events import Event, EventSource, EventStatus, EventType
from courtvision.domain.models import Game
from courtvision.domain.observations import ClockState
from courtvision.scoreboard.state import ScoreboardReading

_EPS = 0.05


class PeriodTracker:
    def __init__(self, game_id: str, game: Game, end_settle_s: float = 4.0) -> None:
        self.game_id = game_id
        self.game = game
        self.end_settle_s = end_settle_s
        self._period: Optional[int] = None
        self._started: set[int] = set()
        self._ended: set[int] = set()
        self._game_started = False
        self._game_ended = False
        self._score: Optional[tuple[int, int]] = None
        self._last: Optional[tuple[float, float, bool]] = None  # (ts, clock, running) der aktuellen Periode
        self._end_ts: dict[int, float] = {}
        self._pending_end: Optional[int] = None  # Periode, ueber deren Spielende noch zu entscheiden ist

    # ------------------------------------------------------------ Zustand
    @property
    def game_started(self) -> bool:
        return self._game_started

    @property
    def game_ended(self) -> bool:
        return self._game_ended

    @property
    def period(self) -> Optional[int]:
        return self._period

    def set_score(self, home: Optional[int], away: Optional[int]) -> None:
        if home is not None and away is not None:
            self._score = (int(home), int(away))

    # ------------------------------------------------------------ Eingang
    def feed(self, item: Union[ClockState, ScoreboardReading]) -> list[Event]:
        if isinstance(item, ScoreboardReading):
            self.set_score(item.home, item.away)
            cs = item.to_clock_state()
            if cs is None:
                return self._decide_end(item.video_ts)
            return self.on_clock(cs.video_ts, cs.period, cs.clock, cs.running)
        return self.on_clock(item.video_ts, item.period, item.clock, item.running)

    def on_clock(self, ts: float, period: int, clock: float, running: bool) -> list[Event]:
        if self._game_ended or period <= 0:
            return []
        out: list[Event] = []
        if self._period is not None and period < self._period:
            return []  # Ausreisser rueckwaerts
        if self._period is not None and period > self._period:
            out += self._close(self._period, ts, estimate=True)
            self._pending_end = None  # Spiel ging weiter
            self._period, self._last = period, None
        elif self._period is None:
            self._period = period
        if running and period not in self._started:
            out += self._start(period, ts, clock)
        if period in self._started and period not in self._ended and clock <= _EPS and not running:
            out += self._close(period, ts, estimate=False, clock_at_zero_ts=ts)
        self._last = (ts, clock, running)
        out += self._decide_end(ts)
        return out

    def start(self, video_ts: float, clock: Optional[float] = None, period: int = 1) -> list[Event]:
        """Expliziter Spielstart (z. B. per Konsole); idempotent."""
        if self._game_ended:
            return []
        if clock is None:
            clock = self.game.period_length_s if period <= self.game.periods else self.game.overtime_length_s
        if self._period is None:
            self._period = period
        return self._start(period, video_ts, clock)

    def end(self, video_ts: Optional[float] = None) -> list[Event]:
        """Expliziter Abschluss: offene Periode beenden und ``GAME_END`` emittieren (idempotent).
        Ohne bisherigen Spielstart passiert nichts."""
        if self._game_ended or not self._game_started or self._period is None:
            return []
        out: list[Event] = []
        p = self._period
        ts = video_ts if video_ts is not None else (self._last[0] if self._last else 0.0)
        if p in self._started and p not in self._ended:
            out += self._close(p, ts, estimate=True)
        out += self._game_end(p, self._end_ts.get(p, ts))
        return out

    # ------------------------------------------------------------ intern
    def _ev(self, typ: EventType, ts: float, period: int, clock: float) -> Event:
        return Event(game_id=self.game_id, type=typ, video_ts=ts, period=period, clock=clock,
                     source=EventSource.SCOREBOARD, status=EventStatus.CONFIRMED)

    def _start(self, period: int, ts: float, clock: float) -> list[Event]:
        out: list[Event] = []
        if period == 1 and not self._game_started:
            self._game_started = True
            out.append(self._ev(EventType.GAME_START, ts, 1, clock))
        if period not in self._started:
            self._started.add(period)
            out.append(self._ev(EventType.PERIOD_START, ts, period, clock))
        if not self._game_started:
            self._game_started = True  # Late-Join mitten im Spiel: kein GAME_START erfinden
        return out

    def _close(self, period: int, ts: float, estimate: bool, clock_at_zero_ts: Optional[float] = None) -> list[Event]:
        if period in self._ended or period not in self._started:
            return []
        self._ended.add(period)
        end_ts = clock_at_zero_ts if clock_at_zero_ts is not None else ts
        if estimate and self._last is not None:
            lts, lclock, lrun = self._last
            end_ts = min(ts, lts + lclock) if lrun else lts
        self._end_ts[period] = end_ts
        if period >= self.game.periods:
            self._pending_end = period
        return [self._ev(EventType.PERIOD_END, end_ts, period, 0.0)]

    def _decide_end(self, now_ts: float) -> list[Event]:
        p = self._pending_end
        if p is None or self._game_ended or self._score is None:
            return []
        if now_ts - self._end_ts.get(p, now_ts) < self.end_settle_s:
            return []
        if self._score[0] == self._score[1]:
            return []  # Verlaengerung folgt (Periodenwechsel loescht _pending_end)
        return self._game_end(p, self._end_ts[p])

    def _game_end(self, period: int, ts: float) -> list[Event]:
        if self._game_ended:
            return []
        self._game_ended = True
        self._pending_end = None
        return [self._ev(EventType.GAME_END, ts, period, 0.0)]
