"""Event-Log: die einzige Wahrheit für alle Statistiken (Event Sourcing).

Regeln
------
* Events werden nie verändert oder gelöscht. Eine Korrektur ist ein neues Event
  mit ``supersedes=<alte id>``; es ersetzt das alte vollständig.
* Ein Event vom Typ ``VOID`` mit ``supersedes`` entfernt das alte Event.
* Eine Bestätigung ist ein neues Event (Kopie) mit ``status=CONFIRMED`` und
  ``supersedes=<alte id>``. Helfer: :func:`confirm`, :func:`correct`, :func:`void`.
* Der Box-Score wird immer aus :func:`effective_events` neu berechnet.

Zeitbasis
---------
* ``video_ts``: Sekunden seit Start der Aufzeichnung (Clip-Referenz).
* ``period``/``clock``: Viertel (1..4, 5+ = Verlängerung) und Restzeit der
  Spieluhr in Sekunden (von der Anzeigetafel; Zeitbasis fürs Play-by-Play).

Payload-Schemata je Typ (alle Schlüssel Pflicht, sofern nicht "optional"):

==================  ==========================================================
SHOT                value: 2|3, made: bool, x: float, y: float (Feldmeter),
                    three_prob: float (optional, P(3er) aus Unsicherheitsband)
FREE_THROW          made: bool, attempt: int, of: int
REBOUND             offensive: bool  (player_id None => Team-Rebound)
ASSIST              shot_id: str
STEAL               victim_id: str|None
BLOCK               shot_id: str, shooter_id: str|None
TURNOVER            kind: TurnoverKind  (player_id None => Team-Turnover)
FOUL                kind: FoulKind, fouled_id: str|None
SUBSTITUTION        player_in: str, player_out: str
STARTERS            players: list[str] (5 player ids of ``team``)
PASS                to_id: str
POSSESSION          start_ts: float, end_ts: float  (Ballbesitzsegment)
JUMP_BALL           winner_team: Side|None
TIMEOUT             {}
SCOREBOARD          home: int, away: int, clock: float, period: int,
                    running: bool, fouls_home: int|None, fouls_away: int|None
PERIOD_START/END    {}
GAME_START/END      {}
SCORESHEET          players: {player_id: {points:int, fouls:int, ftm:int}},
                    home: int, away: int
GAME_FINALIZED      by: str
FLAG                kind: str, message: str, ref: str|None (Hinweis/Abweichung)
VOID                {} (entfernt ``supersedes``)
==================  ==========================================================
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from .models import Side


class EventType(str, Enum):
    GAME_START = "GAME_START"
    GAME_END = "GAME_END"
    PERIOD_START = "PERIOD_START"
    PERIOD_END = "PERIOD_END"
    STARTERS = "STARTERS"
    SHOT = "SHOT"
    FREE_THROW = "FREE_THROW"
    REBOUND = "REBOUND"
    ASSIST = "ASSIST"
    STEAL = "STEAL"
    BLOCK = "BLOCK"
    TURNOVER = "TURNOVER"
    FOUL = "FOUL"
    SUBSTITUTION = "SUBSTITUTION"
    PASS = "PASS"
    POSSESSION = "POSSESSION"
    JUMP_BALL = "JUMP_BALL"
    TIMEOUT = "TIMEOUT"
    SCOREBOARD = "SCOREBOARD"
    SCORESHEET = "SCORESHEET"
    GAME_FINALIZED = "GAME_FINALIZED"
    FLAG = "FLAG"
    VOID = "VOID"


class EventSource(str, Enum):
    OPTICAL = "optical"  # Edge/Cloud CV-Pipeline
    SCOREBOARD = "scoreboard"  # Anzeigetafel
    CONSOLE = "console"  # Kampfgericht-Konsole / Dashboard-Korrektur
    SCORESHEET = "scoresheet"  # offizieller Spielberichtsbogen
    SYSTEM = "system"  # Fusion / Regeln


class EventStatus(str, Enum):
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"


class TurnoverKind(str, Enum):
    BAD_PASS = "bad_pass"
    STEAL = "steal"  # Ballverlust durch Steal
    OUT_OF_BOUNDS = "out_of_bounds"
    TRAVEL = "travel"  # Schrittfehler pauschal
    OFFENSIVE_FOUL = "offensive_foul"
    SHOT_CLOCK = "shot_clock"  # 24 s, Team-Turnover
    OTHER = "other"


class FoulKind(str, Enum):
    PERSONAL = "personal"
    OFFENSIVE = "offensive"
    TECHNICAL = "technical"  # zählt nicht als Teamfoul
    UNSPORTSMANLIKE = "unsportsmanlike"
    DISQUALIFYING = "disqualifying"


#: Ereignistypen, die Spielerstatistiken beeinflussen (für Korrekturliste etc.).
STAT_EVENT_TYPES = frozenset(
    {
        EventType.SHOT,
        EventType.FREE_THROW,
        EventType.REBOUND,
        EventType.ASSIST,
        EventType.STEAL,
        EventType.BLOCK,
        EventType.TURNOVER,
        EventType.FOUL,
        EventType.SUBSTITUTION,
    }
)


class Event(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    game_id: str
    seq: int = 0  # vom Store vergeben, monoton je Spiel
    type: EventType
    video_ts: float = 0.0
    period: int = 0
    clock: Optional[float] = None
    team: Optional[Side] = None
    player_id: Optional[str] = None
    track_id: Optional[int] = None  # falls Spieler noch nicht identifiziert
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source: EventSource = EventSource.OPTICAL
    status: EventStatus = EventStatus.PROVISIONAL
    supersedes: Optional[str] = None
    clip_start: Optional[float] = None  # video_ts
    clip_end: Optional[float] = None
    created_at: float = 0.0  # wall clock (store)

    def derive(self, **changes: Any) -> "Event":
        """Neues Event als Ersatz für dieses (neue id, supersedes=self.id)."""
        data = self.model_dump()
        data.update(id=uuid.uuid4().hex, seq=0, created_at=0.0, supersedes=self.id)
        if "payload" in changes and changes["payload"] is not None:
            merged = dict(self.payload)
            merged.update(changes.pop("payload"))
            data["payload"] = merged
        data.update(changes)
        return Event(**data)


def confirm(ev: Event, source: EventSource = EventSource.CONSOLE, **changes: Any) -> Event:
    return ev.derive(status=EventStatus.CONFIRMED, source=source, confidence=1.0, **changes)


def correct(ev: Event, source: EventSource = EventSource.CONSOLE, **changes: Any) -> Event:
    """Korrektur durch Mensch: gilt als bestätigt."""
    return confirm(ev, source=source, **changes)


def void(ev: Event, source: EventSource = EventSource.CONSOLE, reason: str = "") -> Event:
    return Event(
        game_id=ev.game_id,
        type=EventType.VOID,
        video_ts=ev.video_ts,
        period=ev.period,
        clock=ev.clock,
        supersedes=ev.id,
        source=source,
        status=EventStatus.CONFIRMED,
        payload={"reason": reason} if reason else {},
    )


def effective_events(events: Iterable[Event]) -> list[Event]:
    """Gültige Events: ohne ersetzte Events und ohne VOIDs, sortiert nach Spielzeit.

    Ersatzketten werden aufgelöst (A <- B <- C ergibt nur C). Sortierung:
    ``video_ts``, dann ``seq``. Ein Ersatz-Event übernimmt seinen eigenen
    ``video_ts`` (normalerweise der des Originals).
    """
    evs = list(events)
    superseded = {e.supersedes for e in evs if e.supersedes}
    out = [e for e in evs if e.id not in superseded and e.type != EventType.VOID]
    out.sort(key=lambda e: (e.video_ts, e.seq))
    return out


def game_seconds(period: int, clock: Optional[float], period_length_s: float = 600.0,
                 overtime_length_s: float = 300.0, periods: int = 4) -> float:
    """Abgelaufene Spielzeit in Sekunden seit Spielbeginn (für Minuten, Sortierung)."""
    if period <= 0:
        return 0.0
    if period <= periods:
        before = (period - 1) * period_length_s
        length = period_length_s
    else:
        before = periods * period_length_s + (period - periods - 1) * overtime_length_s
        length = overtime_length_s
    rem = length if clock is None else clock
    return before + (length - rem)
