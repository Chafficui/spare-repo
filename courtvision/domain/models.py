"""Stammdaten: Spiele, Teams, Spieler, Einwilligungen.

Shared contract for all CourtVision modules. Do not change field names without
updating every consumer.
"""
from __future__ import annotations

from enum import Enum, IntFlag
from typing import Literal, Optional

from pydantic import BaseModel, Field

Side = Literal["home", "away"]
SIDES: tuple[Side, Side] = ("home", "away")


def other_side(side: Side) -> Side:
    return "away" if side == "home" else "home"


class Consent(IntFlag):
    """Einwilligungsstufen je Spieler (Recht & Datenschutz).

    NONE bedeutet: keine Einwilligung erteilt. OBJECTED ist der ausdrückliche
    Widerspruch eines Erwachsenen (Opt-out) bzw. fehlende Elterneinwilligung
    bei Minderjährigen: Overlay ohne Namen, keine Highlights, Stats nur als
    Team-Wert.
    """

    NONE = 0
    STATS_INTERNAL = 1
    STATS_PUBLIC = 2
    HIGHLIGHTS = 4
    LIVESTREAM = 8
    TRAINING_DATA = 16
    OBJECTED = 32

    @classmethod
    def adult_default(cls) -> "Consent":
        """Erwachsene: berechtigtes Interesse mit Widerspruchsmöglichkeit."""
        return cls.STATS_INTERNAL | cls.STATS_PUBLIC | cls.HIGHLIGHTS | cls.LIVESTREAM


class Player(BaseModel):
    id: str
    team_id: str
    number: int = Field(ge=0, le=99)
    name: str
    is_minor: bool = False
    consent: int = int(Consent.adult_default())

    def has(self, flag: Consent) -> bool:
        c = Consent(self.consent)
        if Consent.OBJECTED in c:
            return False
        return flag in c


class Team(BaseModel):
    id: str
    name: str
    club: str = ""
    league: str = ""
    age_class: str = "Senioren"  # e.g. "Senioren", "U18"
    jersey_color: str = ""  # hex or name, Startwert fürs Team-Clustering


class GameStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    PROCESSING = "processing"  # Abpfiff, Cloud-Nachverarbeitung läuft
    REVIEW = "review"  # Korrekturliste offen
    FINAL = "final"


class Game(BaseModel):
    id: str
    venue_id: str = ""
    home_team_id: str
    away_team_id: str
    league: str = ""
    age_class: str = "Senioren"
    start_time: str = ""  # ISO 8601
    period_length_s: float = 600.0  # FIBA 4 x 10 min
    overtime_length_s: float = 300.0
    periods: int = 4
    status: GameStatus = GameStatus.SCHEDULED

    def team_id(self, side: Side) -> str:
        return self.home_team_id if side == "home" else self.away_team_id

    def side_of(self, team_id: str) -> Side:
        if team_id == self.home_team_id:
            return "home"
        if team_id == self.away_team_id:
            return "away"
        raise KeyError(team_id)


class Venue(BaseModel):
    id: str
    name: str
    city: str = ""
    scoreboard_source: Literal["data", "second_camera", "main_camera", "none"] = "none"
