"""Play-by-Play-Texte (Deutsch), datenschutzkonform je Betrachter."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ..domain.events import Event, EventStatus, EventType, effective_events
from ..domain.models import Game, Player
from .auth import Principal
from .visibility import can_see_player

TURNOVER_DE = {
    "bad_pass": "Fehlpass", "steal": "Ballverlust durch Steal", "out_of_bounds": "Ball im Aus",
    "travel": "Schrittfehler", "offensive_foul": "Offensivfoul", "shot_clock": "24-Sekunden-Verstoß",
    "other": "Ballverlust",
}
FOUL_DE = {
    "personal": "Persönliches Foul", "offensive": "Offensivfoul",
    "technical": "Technisches Foul", "unsportsmanlike": "Unsportliches Foul",
    "disqualifying": "Disqualifizierendes Foul",
}
#: Typen ohne eigene Zeile im Play-by-Play.
SKIP = frozenset({EventType.SCOREBOARD, EventType.STARTERS, EventType.PASS, EventType.POSSESSION,
                  EventType.FLAG, EventType.SCORESHEET, EventType.VOID, EventType.ASSIST})


def _val(x: Any) -> Any:
    return getattr(x, "value", x)


def surname(name: str) -> str:
    parts = name.split()
    return parts[-1] if parts else name


def fmt_clock(sec: Optional[float]) -> str:
    s = max(0, int(sec or 0))
    return f"{s // 60:02d}:{s % 60:02d}"


class Namer:
    """Spielerbezeichnung "#7 Mueller"; nicht freigegebene Spieler erscheinen als Team."""

    def __init__(self, game: Game, team_names: Mapping[str, str], players: Mapping[str, Player],
                 viewer: Principal) -> None:
        self.game, self.players, self.viewer = game, players, viewer
        self.team_names = dict(team_names)  # Seite -> Name

    def team(self, side: Optional[str]) -> str:
        if side in ("home", "away"):
            return self.team_names.get(side) or ("Heim" if side == "home" else "Gast")
        return "Team"

    def side_of_player(self, pid: str) -> Optional[str]:
        pl = self.players.get(pid)
        if pl is None:
            return None
        try:
            return self.game.side_of(pl.team_id)
        except KeyError:
            return None

    def player(self, pid: Optional[str], side: Optional[str] = None) -> str:
        if not pid:
            return self.team(side)
        pl = self.players.get(pid)
        if pl is None:
            return f"Spieler {pid}" if self.viewer.operational else self.team(side)
        if not can_see_player(self.viewer, pl):
            return self.team(side or self.side_of_player(pid))
        return f"#{pl.number} {surname(pl.name)}"


def period_label(period: int, game: Game) -> str:
    return f"Q{period}" if period <= game.periods else f"OT{period - game.periods}"


def when(ev: Event, game: Game) -> str:
    if ev.period > 0 and ev.clock is not None:
        return f"{period_label(ev.period, game)} {fmt_clock(ev.clock)} – "
    if ev.period > 0:
        return f"{period_label(ev.period, game)} – "
    return ""


def _resolver(all_events: Iterable[Event]):
    alias = {e.supersedes: e.id for e in all_events if e.supersedes}

    def latest(i: str) -> str:
        seen = 0
        while i in alias and seen < 100:
            i, seen = alias[i], seen + 1
        return i

    return latest


def describe(ev: Event, namer: Namer, assists: Mapping[str, Event]) -> Optional[str]:
    t, p = ev.type, ev.payload
    who = namer.player(ev.player_id, ev.team)
    if t == EventType.SHOT:
        made = bool(p.get("made"))
        kind = "Dreier" if int(p.get("value", 2)) == 3 else "Zweier"
        txt = f"{who} {'trifft' if made else 'verfehlt'} {kind}"
        a = assists.get(ev.id)
        if made and a is not None:
            txt += f" (Assist {namer.player(a.player_id, a.team or ev.team)})"
        return txt
    if t == EventType.FREE_THROW:
        n = f" ({p['attempt']}/{p['of']})" if p.get("attempt") and p.get("of") else ""
        return f"{who} {'trifft' if p.get('made') else 'verfehlt'} Freiwurf{n}"
    if t == EventType.REBOUND:
        kind = "Offensiv-Rebound" if p.get("offensive") else "Defensiv-Rebound"
        return f"Team-Rebound {namer.team(ev.team)} ({kind})" if not ev.player_id else f"{who} {kind}"
    if t == EventType.STEAL:
        return f"{who} Ballgewinn (Steal)"
    if t == EventType.BLOCK:
        return f"{who} blockt den Wurf"
    if t == EventType.TURNOVER:
        return f"{who} {TURNOVER_DE.get(str(_val(p.get('kind', 'other'))), 'Ballverlust')}"
    if t == EventType.FOUL:
        return f"{who} {FOUL_DE.get(str(_val(p.get('kind', 'personal'))), 'Foul')}"
    if t == EventType.SUBSTITUTION:
        return (f"Wechsel {namer.team(ev.team)}: {namer.player(p.get('player_in'), ev.team)} "
                f"für {namer.player(p.get('player_out'), ev.team)}")
    if t == EventType.TIMEOUT:
        return f"Auszeit {namer.team(ev.team)}"
    if t == EventType.JUMP_BALL:
        w = p.get("winner_team")
        return f"Sprungball{(': ' + namer.team(w)) if w else ''}"
    if t == EventType.PERIOD_START:
        return f"{period_label(ev.period, namer.game)} beginnt" if ev.period else "Viertel beginnt"
    if t == EventType.PERIOD_END:
        return f"{period_label(ev.period, namer.game)} beendet" if ev.period else "Viertel beendet"
    if t == EventType.GAME_START:
        return "Spielbeginn"
    if t == EventType.GAME_END:
        return "Spielende"
    if t == EventType.GAME_FINALIZED:
        return "Spielbericht freigegeben"
    return None


def playbyplay(all_events: list[Event], game: Game, namer: Namer, viewer: Principal,
               limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Chronologische Liste (aelteste zuerst); ``limit`` behaelt die juengsten Eintraege."""
    from .visibility import event_visible_type
    latest = _resolver(all_events)
    eff = effective_events(all_events)
    assists = {latest(e.payload["shot_id"]): e for e in eff
               if e.type == EventType.ASSIST and e.payload.get("shot_id")}
    out: list[dict[str, Any]] = []
    for ev in eff:
        if ev.type in SKIP or not event_visible_type(viewer, ev):
            continue
        body = describe(ev, namer, assists)
        if body is None:
            continue
        out.append({
            "id": ev.id, "seq": ev.seq, "type": ev.type.value, "period": ev.period, "clock": ev.clock,
            "team": ev.team, "video_ts": ev.video_ts, "text": when(ev, game) + body,
            "provisional": ev.status == EventStatus.PROVISIONAL and ev.confidence < 0.8,
        })
    return out[-limit:] if limit else out
