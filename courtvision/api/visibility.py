"""Sichtbarkeitsregeln (Recht & Datenschutz) fuer Spieler-, Event- und Ratingdaten.

* Betriebsrollen (admin, hallenwart, kampfgericht, system) sehen alles.
* Der Trainer sieht alles ueber Spieler des *eigenen* Teams (``X-Team-Id``).
* Ein Spieler/Elternteil sieht die eigenen Daten (``X-Player-Id``).
* Alle anderen (fan, verband, fremde Trainer/Spieler) sehen Einzelstatistiken nur
  fuer Spieler mit ``STATS_PUBLIC`` und ohne Widerspruch; alle anderen
  Spieler erscheinen nur als Team-Wert (auch im Play-by-Play und Shot Chart).
* Ratings/Noten Minderjaehriger sind nie oeffentlich.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ..domain.events import Event, EventType
from ..domain.models import Player
from ..privacy import can_show_stats_public, public_ratings_allowed
from .auth import Principal


def full_access(p: Principal, team_id: Optional[str]) -> bool:
    """Voller Einblick in Daten des Teams ``team_id``."""
    return p.operational or (p.role == "trainer" and team_id is not None and p.team_id == team_id)


def is_self(p: Principal, player_id: str) -> bool:
    return p.role == "spieler" and p.player_id == player_id


def can_see_player(p: Principal, player: Optional[Player]) -> bool:
    if player is None:
        return p.operational
    return full_access(p, player.team_id) or is_self(p, player.id) or can_show_stats_public(player)


def can_see_rating(p: Principal, player: Optional[Player]) -> bool:
    if player is None:
        return p.operational
    return full_access(p, player.team_id) or is_self(p, player.id) or public_ratings_allowed(player)


def can_see_private_fields(p: Principal, player: Player) -> bool:
    """Minderjaehrigen-Flag, Einwilligungsstufen."""
    return full_access(p, player.team_id) or is_self(p, player.id)


def player_view(p: Principal, pl: Player) -> dict[str, Any]:
    visible = can_see_player(p, pl)
    out: dict[str, Any] = {"id": pl.id, "team_id": pl.team_id, "number": pl.number,
                           "name": pl.name if visible else None}
    if can_see_private_fields(p, pl):
        out.update(is_minor=pl.is_minor, consent=pl.consent)
    return out


#: Ereignistypen, die fuer die Oeffentlichkeit (Fans) bestimmt sind.
PUBLIC_EVENT_TYPES = frozenset({
    EventType.SHOT, EventType.FREE_THROW, EventType.REBOUND, EventType.ASSIST, EventType.STEAL,
    EventType.BLOCK, EventType.TURNOVER, EventType.FOUL, EventType.SUBSTITUTION, EventType.TIMEOUT,
    EventType.SCOREBOARD, EventType.PERIOD_START, EventType.PERIOD_END, EventType.GAME_START,
    EventType.GAME_END, EventType.JUMP_BALL, EventType.GAME_FINALIZED, EventType.VOID,
})
_ID_KEYS = ("fouled_id", "victim_id", "shooter_id", "player_in", "player_out", "to_id")


def event_visible_type(p: Principal, ev: Event) -> bool:
    if p.operational or p.role == "trainer":
        return True
    return ev.type in PUBLIC_EVENT_TYPES


def sanitize_event(ev: Event, p: Principal, players: Mapping[str, Player]) -> Optional[dict[str, Any]]:
    """JSON-Darstellung des Events fuer ``p`` (None: Typ nicht fuer diese Rolle bestimmt)."""
    if not event_visible_type(p, ev):
        return None
    d = ev.model_dump(mode="json")
    if p.operational:
        return d

    def ok(pid: Optional[str]) -> bool:
        return can_see_player(p, players.get(pid)) if pid else True

    if not ok(ev.player_id):
        d["player_id"] = None
        d["track_id"] = None
    pay = d["payload"]
    for k in _ID_KEYS:
        if k in pay and not ok(pay[k]):
            pay[k] = None
    if isinstance(pay.get("players"), list):
        pay["players"] = [x for x in pay["players"] if ok(x)]
    if isinstance(pay.get("candidates"), list):
        pay.pop("candidates")
    return d
