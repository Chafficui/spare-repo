"""Request-Modelle der API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..domain.events import EventSource, EventStatus, EventType, FoulKind
from ..domain.models import Side


class VenueIn(BaseModel):
    id: Optional[str] = None
    name: str
    city: str = ""
    scoreboard_source: Literal["data", "second_camera", "main_camera", "none"] = "none"


class TeamIn(BaseModel):
    id: Optional[str] = None
    name: str
    club: str = ""
    league: str = ""
    age_class: str = "Senioren"
    jersey_color: str = ""


class PlayerIn(BaseModel):
    id: Optional[str] = None
    team_id: str
    number: int = Field(ge=0, le=99)
    name: str
    birth_year: Optional[int] = None
    consent: Optional[str] = None  # wie CSV-Spalte (ja|nein|intern|public|...)


class GameIn(BaseModel):
    id: Optional[str] = None
    venue_id: str = ""
    home_team_id: str
    away_team_id: str
    league: Optional[str] = None
    age_class: Optional[str] = None
    start_time: str = ""
    period_length_s: float = 600.0
    overtime_length_s: float = 300.0
    periods: int = 4


class CalibrationIn(BaseModel):
    #: Landmark-Name -> Klick im Bild [u, v] (Pixel)
    clicks: dict[str, tuple[float, float]]
    #: Zonen als Polygone in Feldkoordinaten (Meter)
    zones: Optional[dict[str, list[tuple[float, float]]]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class StartIn(BaseModel):
    starters: dict[Side, list[str]]
    video_ts: float = 0.0


class EndIn(BaseModel):
    video_ts: Optional[float] = None


class MediaIn(BaseModel):
    video_url: Optional[str] = None
    stream_url: Optional[str] = None


class EventIn(BaseModel):
    id: Optional[str] = None
    type: EventType
    video_ts: float = 0.0
    period: int = 0
    clock: Optional[float] = None
    team: Optional[Side] = None
    player_id: Optional[str] = None
    track_id: Optional[int] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: Optional[EventSource] = None
    status: Optional[EventStatus] = None
    supersedes: Optional[str] = None
    clip_start: Optional[float] = None
    clip_end: Optional[float] = None


class ScoreboardIn(BaseModel):
    video_ts: float
    home: Optional[int] = None
    away: Optional[int] = None
    clock: Optional[float] = None
    period: Optional[int] = None
    running: Optional[bool] = None
    fouls_home: Optional[int] = None
    fouls_away: Optional[int] = None
    conf: dict[str, float] = Field(default_factory=dict)
    source: str = "api"


class ReviewIn(BaseModel):
    action: Literal["confirm", "reject", "suggestion"]
    suggestion_index: Optional[int] = None


class FoulIn(BaseModel):
    player_id: str
    kind: FoulKind = FoulKind.PERSONAL
    fouled_id: Optional[str] = None
    video_ts: Optional[float] = None
    period: Optional[int] = None
    clock: Optional[float] = None


class ScoresheetIn(BaseModel):
    players: dict[str, dict[str, int]] = Field(default_factory=dict)
    home: Optional[int] = None
    away: Optional[int] = None


class FinalizeIn(BaseModel):
    by: str = ""
    force: bool = False


class ConsentIn(BaseModel):
    action: Literal["grant", "revoke", "object", "withdraw_objection", "parental_consent",
                    "revoke_parental", "child_consent", "revoke_child"]
    #: STATS_INTERNAL | STATS_PUBLIC | HIGHLIGHTS | LIVESTREAM | TRAINING_DATA (grant/revoke/parental_consent)
    level: Optional[list[str]] = None
    note: str = ""


class DpiaUpdate(BaseModel):
    status: Literal["open", "in_progress", "done"]
    owner: str = ""
    note: str = ""
