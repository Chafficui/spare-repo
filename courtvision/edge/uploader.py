"""Resumable, persistente Upload-Queue fuer Videosegmente.

* Segmente laufen ueber die Zustaende ``pending -> uploading -> done``; der Zustand
  steht in SQLite und uebersteht Neustarts (``uploading`` wird beim Start wieder
  ``pending``, der Server-Offset entscheidet ueber das Fortsetzen).
* Chunked Upload mit Offset: nach Netzfehlern wird mit exponentiellem Backoff
  (``base * 2**(n-1)``, gedeckelt) erneut versucht; die Datei bleibt lokal.
* Offline (``is_online() == False``): nichts passiert, kein Fehlversuch gezaehlt.
* Hochgeladen wird schon WAEHREND des Spiels (neue Segmente werden eingereiht,
  sobald sie abgeschlossen sind), damit der volle Box-Score <= 15 min nach dem
  Abpfiff fertig sein kann (Plan: Latenzbudget). Reihenfolge: aelteste zuerst.
* Transport ist injizierbar (:class:`Transport`); Tests nutzen ein Fake.

LTE-Fallback
------------
Hallen ohne nutzbares Netz bekommen einen LTE-Router (Plan: Ausfallsicherheit).
Mobilfunk ist gedeckelt und teuer: :class:`LteFallback` ist eine Link-Policy, die
auf LTE nur kleine Daten (Events, Live-Score) und optional ein Datenbudget fuer
Segmente zulaesst; volle Rohsegmente warten standardmaessig auf einen
unbegrenzten Uplink. Ein 1080p30-H.264-Stream braucht 5-14 Mbit/s Upload (YouTube).
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol

#: YouTube-Empfehlung 1080p30, H.264, Mbit/s Upload (Plan: Ausfallsicherheit)
STREAM_1080P30_MBPS = (5.0, 14.0)


class SegmentState(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    DONE = "done"


class TransportError(Exception):
    """Netzwerkfehler/Timeout: spaeter erneut versuchen."""


class Transport(Protocol):
    def begin(self, key: str, size: int) -> str:
        """Startet/ereuert eine Upload-Sitzung, gibt die upload_id zurueck."""

    def committed_offset(self, upload_id: str) -> int:
        """Bytes, die der Server sicher hat (Basis fuers Fortsetzen)."""

    def put_chunk(self, upload_id: str, offset: int, data: bytes) -> None:
        """Haengt ``data`` ab ``offset`` an. Wirft :class:`TransportError` bei Fehlern."""

    def complete(self, upload_id: str) -> None:
        """Schliesst den Upload ab."""


class LinkPolicy(Protocol):
    def allow(self, size_bytes: int) -> bool: ...
    def record(self, size_bytes: int) -> None: ...


@dataclass
class LteFallback:
    """Link-Policy fuer LTE: Datenbudget; Rohsegmente nur, wenn ``allow_segments``."""

    budget_bytes: int = 0
    allow_segments: bool = False
    used_bytes: int = 0

    def allow(self, size_bytes: int) -> bool:
        return self.allow_segments and self.used_bytes + size_bytes <= self.budget_bytes

    def record(self, size_bytes: int) -> None:
        self.used_bytes += size_bytes

    def allow_small(self, size_bytes: int) -> bool:
        """Events/Live-Score (wenige KB) gehen immer, solange irgendein Budget bleibt."""
        return size_bytes <= 64 * 1024 and self.used_bytes + size_bytes <= max(self.budget_bytes, 50 * 1024**2)


@dataclass
class QueueItem:
    path: str
    key: str
    size: int
    state: SegmentState
    offset: int
    upload_id: Optional[str]
    attempts: int
    next_attempt: float
    error: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    path TEXT PRIMARY KEY, key TEXT NOT NULL, size INTEGER NOT NULL, state TEXT NOT NULL,
    off INTEGER NOT NULL DEFAULT 0, upload_id TEXT, attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '', added_at REAL NOT NULL
);
"""


class UploadQueue:
    def __init__(self, db_path: str = ":memory:", clock: Callable[[], float] = time.time) -> None:
        self._db = sqlite3.connect(db_path)
        self._db.executescript(_SCHEMA)
        self._clock = clock
        # Neustart: halbfertige Uploads wieder aufnehmbar machen
        self._db.execute("UPDATE segments SET state=? WHERE state=?",
                         (SegmentState.PENDING.value, SegmentState.UPLOADING.value))
        self._db.commit()

    @staticmethod
    def _item(r) -> QueueItem:
        return QueueItem(r[0], r[1], r[2], SegmentState(r[3]), r[4], r[5], r[6], r[7], r[8])

    _COLS = "path, key, size, state, off, upload_id, attempts, next_attempt, error"

    def enqueue(self, path: str, key: Optional[str] = None, size: Optional[int] = None) -> bool:
        """Reiht ein Segment ein (idempotent). Gibt False zurueck, wenn schon bekannt."""
        size = os.path.getsize(path) if size is None else size
        cur = self._db.execute(
            "INSERT OR IGNORE INTO segments (path, key, size, state, added_at) VALUES (?,?,?,?,?)",
            (path, key or os.path.basename(path), size, SegmentState.PENDING.value, self._clock()))
        self._db.commit()
        return cur.rowcount > 0

    def get(self, path: str) -> Optional[QueueItem]:
        r = self._db.execute(f"SELECT {self._COLS} FROM segments WHERE path=?", (path,)).fetchone()
        return self._item(r) if r else None

    def items(self, state: Optional[SegmentState] = None) -> list[QueueItem]:
        q = f"SELECT {self._COLS} FROM segments"
        args: tuple = ()
        if state is not None:
            q += " WHERE state=?"
            args = (state.value,)
        return [self._item(r) for r in self._db.execute(q + " ORDER BY added_at, path", args)]

    def next_due(self, now: Optional[float] = None) -> Optional[QueueItem]:
        now = self._clock() if now is None else now
        r = self._db.execute(
            f"SELECT {self._COLS} FROM segments WHERE state=? AND next_attempt<=? "
            f"ORDER BY added_at, path LIMIT 1", (SegmentState.PENDING.value, now)).fetchone()
        return self._item(r) if r else None

    def update(self, path: str, **cols) -> None:
        names = ", ".join(f"{'off' if k == 'offset' else k}=?" for k in cols)
        vals = [v.value if isinstance(v, SegmentState) else v for v in cols.values()]
        self._db.execute(f"UPDATE segments SET {names} WHERE path=?", (*vals, path))
        self._db.commit()

    def pending_bytes(self) -> int:
        r = self._db.execute("SELECT COALESCE(SUM(size-off),0) FROM segments WHERE state!=?",
                             (SegmentState.DONE.value,)).fetchone()
        return int(r[0])


def backoff_s(attempts: int, base: float = 2.0, cap: float = 300.0) -> float:
    """Wartezeit nach dem ``attempts``-ten Fehlschlag: base, 2*base, 4*base ... bis cap."""
    return min(cap, base * 2 ** max(0, attempts - 1))


class Uploader:
    def __init__(self, queue: UploadQueue, transport: Transport, *, chunk_size: int = 4 * 1024**2,
                 is_online: Callable[[], bool] = lambda: True, link: Optional[LinkPolicy] = None,
                 on_done: Optional[Callable[[QueueItem], None]] = None,
                 clock: Callable[[], float] = time.time, backoff_base: float = 2.0,
                 backoff_cap: float = 300.0) -> None:
        self.queue, self.transport, self.chunk_size = queue, transport, chunk_size
        self.is_online, self.link, self.on_done = is_online, link, on_done
        self._clock = clock
        self.backoff_base, self.backoff_cap = backoff_base, backoff_cap

    def _fail(self, item: QueueItem, err: Exception) -> None:
        n = item.attempts + 1
        self.queue.update(item.path, state=SegmentState.PENDING, attempts=n, error=str(err),
                          next_attempt=self._clock() + backoff_s(n, self.backoff_base, self.backoff_cap))

    def upload_one(self, item: QueueItem) -> bool:
        """Laedt ein Segment (fortsetzend) hoch. True, wenn fertig; sonst Backoff eingetragen."""
        if self.link is not None and not self.link.allow(item.size - item.offset):
            return False  # Policy (LTE): spaeter, kein Fehlversuch
        self.queue.update(item.path, state=SegmentState.UPLOADING)
        try:
            uid = item.upload_id or self.transport.begin(item.key, item.size)
            self.queue.update(item.path, upload_id=uid)
            offset = min(self.transport.committed_offset(uid), item.size)  # Server ist massgeblich
            with open(item.path, "rb") as f:
                while offset < item.size:
                    f.seek(offset)
                    data = f.read(min(self.chunk_size, item.size - offset))
                    if not data:
                        raise TransportError("Datei kuerzer als erwartet")
                    self.transport.put_chunk(uid, offset, data)
                    offset += len(data)
                    self.queue.update(item.path, offset=offset)
                    if self.link is not None:
                        self.link.record(len(data))
            self.transport.complete(uid)
        except (TransportError, OSError) as e:
            self._fail(self.queue.get(item.path) or item, e)
            return False
        self.queue.update(item.path, state=SegmentState.DONE, offset=item.size, error="")
        if self.on_done is not None:
            self.on_done(self.queue.get(item.path))
        return True

    def run_once(self) -> dict[str, int]:
        """Arbeitet alle faelligen Segmente ab (aelteste zuerst). Bei einem Fehler wird die
        Runde beendet (der Link ist vermutlich weg); der Backoff steht in der Queue."""
        res = {"done": 0, "failed": 0, "skipped_offline": 0}
        while True:
            if not self.is_online():
                res["skipped_offline"] += 1
                return res
            item = self.queue.next_due(self._clock())
            if item is None:
                return res
            if self.upload_one(item):
                res["done"] += 1
            else:
                res["failed"] += 1
                return res


# ---------------------------------------------------------------------------
# Bandbreite
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamFeasibility:
    feasible: bool
    usable_mbps: float
    recommended_bitrate_mbps: float
    reason: str = ""

    def __bool__(self) -> bool:
        return self.feasible


def stream_feasible(upload_mbps: float, headroom: float = 0.8) -> StreamFeasibility:
    """Reicht der Upload fuer einen 1080p30-H.264-Stream (YouTube: 5-14 Mbit/s)?

    Nutzbar sind nur ``headroom`` (80 %) des gemessenen Uplinks. Empfohlene Bitrate:
    so hoch wie moeglich, hoechstens 14 Mbit/s.
    """
    lo, hi = STREAM_1080P30_MBPS
    usable = max(0.0, upload_mbps) * headroom
    if usable < lo:
        return StreamFeasibility(False, usable, 0.0,
                                 f"nutzbarer Upload {usable:.1f} Mbit/s < {lo} Mbit/s Minimum")
    return StreamFeasibility(True, usable, min(hi, usable))


def realtime_upload_possible(upload_mbps: float, stream_mbps: float, record_mbps: float,
                             headroom: float = 0.8) -> bool:
    """Bleibt neben dem Livestream genug Uplink, um die Segmente (Aufnahmebitrate) in
    Echtzeit hochzuladen? Sonst wird nach dem Spiel nachgeladen."""
    return upload_mbps * headroom - stream_mbps >= record_mbps


def upload_backlog_eta_s(pending_bytes: int, upload_mbps: float, headroom: float = 0.8) -> float:
    """Sekunden bis der Rueckstand hochgeladen ist (Budget: Box-Score <= 15 min nach Abpfiff)."""
    rate = upload_mbps * headroom * 1e6 / 8
    return float("inf") if rate <= 0 else pending_bytes / rate
