"""Append-only Event-Store (SQLite; Schema PostgreSQL-kompatibel gehalten).

Alle Oberflächen lesen aus demselben Event-Log. Korrekturen sind neue Events.
Listener (z. B. WebSocket-Broadcast) werden nach jedem Append synchron
aufgerufen.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Callable, Iterable, Optional

from ..domain.events import Event, EventType, effective_events
from ..domain.models import Game, GameStatus, Player, Team, Venue

Listener = Callable[[Event], None]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS venues (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS teams (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS players (id TEXT PRIMARY KEY, team_id TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    UNIQUE (game_id, seq)
);
CREATE INDEX IF NOT EXISTS events_game ON events (game_id, seq);
CREATE TABLE IF NOT EXISTS docs (
    kind TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (kind, id)
);
"""


class EventStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []

    def _all(self, sql: str, args: tuple = ()) -> list:
        """Lesezugriffe unter der Sperre: eine SQLite-Verbindung ist nicht fuer gleichzeitige Cursor mehrerer Threads
        gedacht (``InterfaceError: bad parameter or other API misuse``, z. B. Polling waehrend eine Simulation schreibt)."""
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _one(self, sql: str, args: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    # ------------------------------------------------------------ listeners
    def subscribe(self, fn: Listener) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    # ------------------------------------------------------------ master data
    def _put(self, table: str, obj, extra: Optional[dict] = None) -> None:
        cols = {"id": obj.id, "data": obj.model_dump_json()}
        if extra:
            cols.update(extra)
        names = ", ".join(cols)
        marks = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{k}=excluded.{k}" for k in cols if k != "id")
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {table} ({names}) VALUES ({marks}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                tuple(cols.values()),
            )
            self._conn.commit()

    def _get(self, table: str, model, id_: str):
        row = self._one(f"SELECT data FROM {table} WHERE id=?", (id_,))
        return model.model_validate_json(row[0]) if row else None

    def put_venue(self, v: Venue) -> None:
        self._put("venues", v)

    def get_venue(self, id_: str) -> Optional[Venue]:
        return self._get("venues", Venue, id_)

    def put_team(self, t: Team) -> None:
        self._put("teams", t)

    def get_team(self, id_: str) -> Optional[Team]:
        return self._get("teams", Team, id_)

    def list_teams(self) -> list[Team]:
        return [Team.model_validate_json(r[0]) for r in self._all("SELECT data FROM teams")]

    def put_player(self, p: Player) -> None:
        self._put("players", p, {"team_id": p.team_id})

    def get_player(self, id_: str) -> Optional[Player]:
        return self._get("players", Player, id_)

    def roster(self, team_id: str) -> list[Player]:
        rows = self._all("SELECT data FROM players WHERE team_id=?", (team_id,))
        return sorted((Player.model_validate_json(r[0]) for r in rows), key=lambda p: p.number)

    def list_venues(self) -> list[Venue]:
        return [Venue.model_validate_json(r[0]) for r in self._all("SELECT data FROM venues")]

    # ------------------------------------------------------------ documents
    def put_doc(self, kind: str, id_: str, data) -> None:
        """Beliebiges JSON-Dokument (Kalibrierung, Medien-URLs, Einwilligungs-Audit ...)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO docs (kind, id, data) VALUES (?, ?, ?) "
                "ON CONFLICT(kind, id) DO UPDATE SET data=excluded.data",
                (kind, id_, json.dumps(data)),
            )
            self._conn.commit()

    def get_doc(self, kind: str, id_: str, default=None):
        row = self._one("SELECT data FROM docs WHERE kind=? AND id=?", (kind, id_))
        return json.loads(row[0]) if row else default

    def delete_doc(self, kind: str, id_: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM docs WHERE kind=? AND id=?", (kind, id_))
            self._conn.commit()
            return cur.rowcount > 0

    def list_docs(self, kind: str) -> dict:
        return {r[0]: json.loads(r[1]) for r in self._all("SELECT id, data FROM docs WHERE kind=?", (kind,))}

    def put_game(self, g: Game) -> None:
        self._put("games", g)

    def get_game(self, id_: str) -> Optional[Game]:
        return self._get("games", Game, id_)

    def list_games(self) -> list[Game]:
        return [Game.model_validate_json(r[0]) for r in self._all("SELECT data FROM games")]

    def set_game_status(self, id_: str, status: GameStatus) -> Game:
        g = self.get_game(id_)
        if g is None:
            raise KeyError(id_)
        g = g.model_copy(update={"status": status})
        self.put_game(g)
        return g

    # ------------------------------------------------------------ events
    def append(self, ev: Event) -> Event:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id=?", (ev.game_id,)
            ).fetchone()
            ev = ev.model_copy(update={"seq": row[0] + 1, "created_at": time.time()})
            self._conn.execute(
                "INSERT INTO events (id, game_id, seq, type, data) VALUES (?, ?, ?, ?, ?)",
                (ev.id, ev.game_id, ev.seq, ev.type.value, ev.model_dump_json()),
            )
            self._conn.commit()
        for fn in list(self._listeners):
            try:
                fn(ev)
            except Exception:  # Listener-Fehler dürfen das Log nie blockieren
                pass
        return ev

    def append_many(self, evs: Iterable[Event]) -> list[Event]:
        return [self.append(e) for e in evs]

    def events(self, game_id: str, after_seq: int = 0, type_: Optional[EventType] = None) -> list[Event]:
        q = "SELECT data FROM events WHERE game_id=? AND seq>?"
        args: list = [game_id, after_seq]
        if type_ is not None:
            q += " AND type=?"
            args.append(type_.value)
        q += " ORDER BY seq"
        return [Event.model_validate_json(r[0]) for r in self._all(q, tuple(args))]

    def get_event(self, id_: str) -> Optional[Event]:
        row = self._one("SELECT data FROM events WHERE id=?", (id_,))
        return Event.model_validate_json(row[0]) if row else None

    def effective(self, game_id: str) -> list[Event]:
        return effective_events(self.events(game_id))
