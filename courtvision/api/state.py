"""Gemeinsamer Anwendungszustand der API (Store, Hub, Fusion, Einwilligungen ...)."""
from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from fastapi import HTTPException

from ..domain.events import Event, EventType
from ..domain.models import Consent, Game, GameStatus, Player, Team
from ..geometry.court import Calibration
from ..privacy import ConsentManager, default_checklist
from ..scoreboard import PlausibilityFilter, ScoreboardFusion
from ..stats import BoxScore, LeagueDistribution, PlayerLine, compute_boxscore, game_score, grade_boxscore, Grade
from ..store.eventstore import EventStore
from .csvio import age_from_birth_year, is_minor_by_year, parse_consent


class Subscriber:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[Event] = asyncio.Queue()


class Hub:
    """Verteilt Store-Events thread-sicher an WebSocket-Verbindungen."""

    def __init__(self) -> None:
        self._subs: dict[str, set[Subscriber]] = defaultdict(set)
        self._lock = threading.Lock()

    def add(self, game_id: str, sub: Subscriber) -> None:
        with self._lock:
            self._subs[game_id].add(sub)

    def remove(self, game_id: str, sub: Subscriber) -> None:
        with self._lock:
            self._subs[game_id].discard(sub)

    def on_event(self, ev: Event) -> None:
        """Store-Listener; wird in beliebigen Threads aufgerufen."""
        with self._lock:
            subs = list(self._subs.get(ev.game_id, ()))
        for s in subs:
            try:
                s.loop.call_soon_threadsafe(s.queue.put_nowait, ev)
            except RuntimeError:  # Loop bereits geschlossen
                self.remove(ev.game_id, s)


@dataclass
class ScoreboardPipe:
    filter: PlausibilityFilter
    fusion: ScoreboardFusion
    lock: threading.Lock = field(default_factory=threading.Lock)


class AppState:
    def __init__(self, store: EventStore, scoreboard_n_stable: int = 3) -> None:
        self.store = store
        self.hub = Hub()
        self.lock = threading.RLock()
        self.cm = ConsentManager()
        self.dpia = default_checklist()
        self.n_stable = scoreboard_n_stable
        self.calibrations: dict[str, dict[str, Any]] = {}
        self.media: dict[str, dict[str, Optional[str]]] = {}
        self._pipes: dict[str, ScoreboardPipe] = {}
        self._box_cache: dict[str, tuple[tuple, BoxScore]] = {}
        store.subscribe(self.hub.on_event)

    # ------------------------------------------------------------ lookups
    def game(self, game_id: str) -> Game:
        g = self.store.get_game(game_id)
        if g is None:
            raise HTTPException(404, f"Spiel '{game_id}' nicht gefunden")
        return g

    def team(self, team_id: str) -> Team:
        t = self.store.get_team(team_id)
        if t is None:
            raise HTTPException(404, f"Team '{team_id}' nicht gefunden")
        return t

    def player(self, player_id: str) -> Player:
        p = self.store.get_player(player_id)
        if p is None:
            raise HTTPException(404, f"Spieler '{player_id}' nicht gefunden")
        return p

    def rosters(self, game: Game) -> dict[str, list[Player]]:
        return {"home": self.store.roster(game.home_team_id), "away": self.store.roster(game.away_team_id)}

    def players_map(self, game: Game) -> dict[str, Player]:
        r = self.rosters(game)
        return {p.id: p for p in r["home"] + r["away"]}

    def team_names(self, game: Game) -> dict[str, str]:
        out = {}
        for side in ("home", "away"):
            t = self.store.get_team(game.team_id(side))  # type: ignore[arg-type]
            out[side] = t.name if t else ("Heim" if side == "home" else "Gast")
        return out

    def box(self, game: Game, events: Optional[list[Event]] = None) -> BoxScore:
        events = self.store.events(game.id) if events is None else events
        r = self.rosters(game)
        key = (events[-1].seq if events else 0, len(r["home"]), len(r["away"]), game.status.value)
        hit = self._box_cache.get(game.id)
        if hit and hit[0] == key:
            return hit[1]
        box = compute_boxscore(events, game, r)
        self._box_cache[game.id] = (key, box)
        return box

    # ------------------------------------------------------- scoreboard pipe
    def pipe(self, game_id: str) -> ScoreboardPipe:
        with self.lock:
            if game_id not in self._pipes:
                self._pipes[game_id] = ScoreboardPipe(PlausibilityFilter(n_stable=self.n_stable),
                                                      ScoreboardFusion(game_id))
            return self._pipes[game_id]

    # -------------------------------------------------------------- consent
    def ensure_registered(self, player_id: str) -> Player:
        """Spieler im ConsentManager anmelden (Zustand aus dem gespeicherten Consent-Wert)."""
        with self.lock:
            if player_id in self.cm.players:
                return self.cm.players[player_id]
            p = self.player(player_id)
            orig = Consent(p.consent)
            self.cm.register(p)
            prof = self.cm._profiles[player_id]  # noqa: SLF001 (Wiederherstellung nach Neustart)
            if Consent.OBJECTED in orig:
                if not p.is_minor:
                    prof.objected = True
                    prof.granted = int(orig & ~Consent.OBJECTED) or int(Consent.adult_default())
            else:
                prof.granted = int(orig)
                if p.is_minor:
                    prof.parental_consent = prof.child_consent = True
            self.cm._sync(player_id)  # noqa: SLF001
            self.store.put_player(p)
            return p

    def persist_player(self, player_id: str) -> Player:
        p = self.cm.players[player_id]
        self.store.put_player(p)
        return p

    def apply_roster_consent(self, player_id: str, raw: str, birth_year: Optional[int]) -> None:
        """CSV-Spalte ``consent`` auf den ConsentManager anwenden (Aktor: Verein)."""
        kind, levels = parse_consent(raw)
        p = self.ensure_registered(player_id)
        if birth_year is not None:
            self.cm.set_age(player_id, age_from_birth_year(birth_year), actor="club")
        note = "CSV-Import (Verein bestätigt Einwilligung)"
        if kind == "object":
            self.cm.object_(player_id, actor="club", note=note)
        elif kind == "grant":
            if p.is_minor:
                self.cm.grant_parental(player_id, levels, note=note)
                if self.cm.needs_child_consent(player_id):
                    self.cm.grant_child(player_id, note=note)
            else:
                for lv in Consent.adult_default() & ~levels:
                    self.cm.revoke(player_id, lv, actor="club", note=note)
                self.cm.grant(player_id, levels, actor="club", note=note)
        self.persist_player(player_id)

    def upsert_player(self, team_id: str, number: int, name: str, birth_year: Optional[int],
                      consent: str, player_id: Optional[str] = None) -> tuple[Player, bool]:
        pid = player_id or f"{team_id}-{number}"
        existing = self.store.get_player(pid)
        minor = is_minor_by_year(birth_year) if birth_year is not None else (existing.is_minor if existing else False)
        with self.lock:
            if existing is None:
                p = Player(id=pid, team_id=team_id, number=number, name=name, is_minor=minor)
                self.cm.register(p)
                self.store.put_player(p)
                created = True
            else:
                p = self.ensure_registered(pid)
                p.number, p.name = number, name
                if p.is_minor != minor:
                    p.is_minor = minor
                    self.cm._sync(pid)  # noqa: SLF001
                self.store.put_player(p)
                created = False
            self.apply_roster_consent(pid, consent, birth_year)
            return self.store.get_player(pid), created  # type: ignore[return-value]

    # ------------------------------------------------------------ statistics
    def league_games(self, league: str, final_only: bool = False) -> list[Game]:
        gs = [g for g in self.store.list_games() if g.league == league and g.status != GameStatus.SCHEDULED
              and (not final_only or g.status == GameStatus.FINAL)]
        gs.sort(key=lambda g: (g.start_time, g.id))
        return gs

    def team_games(self, team_id: str) -> list[Game]:
        gs = [g for g in self.store.list_games() if team_id in (g.home_team_id, g.away_team_id)
              and g.status != GameStatus.SCHEDULED]
        gs.sort(key=lambda g: (g.start_time, g.id))
        return gs

    def collect(self, games: list[Game], dist_games: Optional[list[Game]] = None
                ) -> dict[str, list[tuple[Game, PlayerLine, Grade]]]:
        """Pro Spieler chronologisch (Spiel, Zeile, Note) ueber ``games``; Verteilung aus ``dist_games``."""
        allg = {g.id: g for g in list(dist_games or games) + list(games)}
        boxes = {gid: self.box(g) for gid, g in allg.items()}
        dist = LeagueDistribution()
        for gid, g in allg.items():
            d = _game_date(g)
            for line in boxes[gid].players.values():
                if line.seconds_played >= 300:
                    dist.add(d, game_score(line) * 2400.0 / line.seconds_played)
        out: dict[str, list[tuple[Game, PlayerLine, Grade]]] = defaultdict(list)
        for g in games:
            grades = grade_boxscore(boxes[g.id], dist)
            for pid, line in boxes[g.id].players.items():
                if pid in grades:
                    out[pid].append((g, line, grades[pid]))
        return out


def _game_date(g: Game) -> date:
    try:
        return date.fromisoformat(g.start_time[:10])
    except ValueError:
        return date.today()
