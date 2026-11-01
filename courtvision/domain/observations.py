"""Schnittstelle CV-Pipeline -> Event-Erkennung.

Die CV-Pipeline (``courtvision.cv``) erzeugt einen zeitlich sortierten Strom
dieser Beobachtungen. Der Zustandsautomat (``courtvision.scoring``) konsumiert
ihn und schreibt Events. Alle Koordinaten sind Feldkoordinaten in Metern
(siehe ``courtvision.geometry.court``). ``team`` ist "home"/"away".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from .models import Side

Basket = Literal["left", "right"]  # left: x<0, right: x>0


@dataclass(frozen=True)
class PlayerState:
    track_id: int
    team: Optional[Side]  # None => Schiri/unklar
    x: float
    y: float
    player_id: Optional[str] = None  # None bis Nummer sicher gelesen
    id_confidence: float = 0.0


@dataclass(frozen=True)
class FrameState:
    """Momentaufnahme ca. 10 Hz: alle Tracks im Bild."""

    video_ts: float
    players: tuple[PlayerState, ...]
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    ball_height: Optional[float] = None  # m über Boden, falls geschätzt


@dataclass(frozen=True)
class Possession:
    """Ballbesitz-Signal (10 Hz). track_id None => Ball frei (Pass/Wurf/lose)."""

    video_ts: float
    track_id: Optional[int]
    team: Optional[Side]
    player_id: Optional[str] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class ShotRelease:
    """Ball verlässt Hand mit Aufwärtsbahn Richtung Korb (Parabel-Fit)."""

    video_ts: float
    track_id: int
    team: Side
    x: float  # Fußposition Werfer
    y: float
    basket: Basket
    player_id: Optional[str] = None
    three_prob: float = 0.0  # P(Dreier) aus Lage zur Dreierlinie + Unsicherheit
    confidence: float = 1.0


@dataclass(frozen=True)
class RimMake:
    """Ball durchquert Ringebene von oben nach unten (Treffer-Kandidat)."""

    video_ts: float
    basket: Basket
    confidence: float = 1.0


@dataclass(frozen=True)
class BallTouch:
    """Erkannte Ballberührung durch einen Spieler ohne Kontrolle
    (Verteidigerhand am Ball, abrupter Bahnabbruch). Basis für Block/Steal."""

    video_ts: float
    track_id: int
    team: Side
    player_id: Optional[str] = None
    abrupt_trajectory_change: bool = False
    confidence: float = 1.0


@dataclass(frozen=True)
class OutOfBounds:
    """Ballprojektion ausserhalb des Feldpolygons + Spielunterbrechung."""

    video_ts: float
    last_touch_track: Optional[int]
    last_touch_team: Optional[Side]
    last_touch_player: Optional[str] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class Formation:
    """Formations-Detektor: Freiwurf-Aufstellung oder Sprungball."""

    video_ts: float
    kind: Literal["free_throw", "jump_ball"]
    shooter_track: Optional[int] = None
    shooter_team: Optional[Side] = None
    shooter_player: Optional[str] = None
    basket: Optional[Basket] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class ClockState:
    """Uhrzustand von der Anzeigetafel (für Wechselfenster, Zeitbasis)."""

    video_ts: float
    period: int
    clock: float
    running: bool


Observation = Union[
    FrameState, Possession, ShotRelease, RimMake, BallTouch, OutOfBounds, Formation, ClockState
]


def obs_ts(o: Observation) -> float:
    return o.video_ts
