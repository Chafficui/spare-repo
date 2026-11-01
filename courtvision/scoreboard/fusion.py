"""Fusion von Anzeigetafel und optischen Events (Plan: "Fusion und Korrektur")."""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional, Union

from courtvision.domain.events import (Event, EventSource, EventStatus, EventType,
                                       FoulKind, void)
from courtvision.domain.models import SIDES, Side

from .state import ScoreboardReading


@dataclass
class FusionConfig:
    window_before: float = 10.0  # optisches Event bis 10 s VOR der Tafeländerung
    window_after: float = 5.0  # ... bis 5 s danach
    miss_search_before: float = 20.0  # Rückwärtssuche nach Fehlwurf
    confirmed_conf: float = 0.95
    missed_make_conf: float = 0.4
    new_shot_conf: float = 0.3
    expired_make_conf: float = 0.6
    default_lag: float = 2.0  # neu angelegter Wurf liegt so viel vor der Tafeländerung
    max_jump: int = 12
    scoreboard_min_interval: float = 5.0  # Uhr-only-Änderungen: höchstens alle x s
    board_stale_s: float = 5.0  # Ablauf-Prüfung nur, wenn Tafel zuletzt so frisch war
    record_ttl: float = 180.0


@dataclass
class _Rec:
    ev: Event
    consumed: bool = False


def _made(ev: Event) -> bool:
    return bool(ev.payload.get("made"))


class ScoreboardFusion:
    def __init__(self, game_id: str, config: Optional[FusionConfig] = None):
        self.game_id = game_id
        self.cfg = config or FusionConfig()
        self._recs: dict[str, _Rec] = {}
        self._state = ScoreboardReading(video_ts=0.0)
        self._score: dict[Side, int] = {}
        self._fouls: dict[Side, Optional[int]] = {}
        self._timeline: list[tuple[float, int, float, bool]] = []
        self._last_ts: Optional[float] = None
        self._last_key: Optional[tuple] = None
        self._last_emit_ts = -1e9
        self._last_clock: Optional[float] = None
        self._resolved: list[dict] = []

    # ------------------------------------------------------------ Zeitbasis
    def clock_at(self, video_ts: float) -> Optional[tuple[int, float]]:
        """(Viertel, Restzeit) zum Zeitpunkt video_ts aus den bisherigen Tafelwerten."""
        if not self._timeline:
            return None
        i = bisect.bisect_right([t[0] for t in self._timeline], video_ts) - 1
        t, period, clock, running = self._timeline[max(i, 0)]
        if running and video_ts > t:
            clock = max(0.0, clock - (video_ts - t))
        return period, clock

    def _stamp(self, ts: float) -> dict:
        pc = self.clock_at(ts)
        return {} if pc is None else {"period": pc[0], "clock": pc[1]}

    # --------------------------------------------------------------- Eingang
    def on_optical_event(self, ev: Event) -> list[Event]:
        if ev.supersedes and ev.supersedes in self._recs:
            rec = self._recs.pop(ev.supersedes)
            rec.ev = ev
            self._recs[ev.id] = rec
            return []
        if ev.type not in (EventType.SHOT, EventType.FREE_THROW, EventType.FOUL) or ev.team is None:
            return []
        rec = _Rec(ev)
        self._recs[ev.id] = rec
        out: list[Event] = []
        if ev.type != EventType.FOUL and _made(ev):
            # Tafel war schneller als die Optik: zuvor synthetisch angelegten Wurf ersetzen
            for r in self._resolved:
                if r["kind"] != "created" or r["side"] != ev.team:
                    continue
                if (ev.type == EventType.FREE_THROW) != (r["part"] == 1):
                    continue
                if not (r["ts"] - self.cfg.window_before <= ev.video_ts <= r["ts"] + self.cfg.window_after):
                    continue
                created = self._recs.pop(r["id"], None)
                if created is None:
                    continue
                out.append(void(created.ev, EventSource.SYSTEM, "durch optisches Event ersetzt"))
                self._resolved.remove(r)
                rec.consumed = True
                out.append(self._confirm(rec, r["part"], r["ts"]))
                break
        return out

    def on_reading(self, reading: ScoreboardReading) -> list[Event]:
        out: list[Event] = []
        ts = reading.video_ts
        self._last_ts = ts
        s = self._state
        period_changed = (reading.period is not None and s.period is not None
                          and reading.period != s.period)
        for n in ("home", "away", "clock", "period", "running", "fouls_home", "fouls_away"):
            v = getattr(reading, n)
            if v is not None:
                setattr(s, n, v)
                if n in reading.conf:
                    s.conf[n] = reading.conf[n]
        s.video_ts, s.source = ts, reading.source
        if s.period is not None and s.clock is not None:
            self._timeline.append((ts, int(s.period), float(s.clock), bool(s.running)))
            del self._timeline[:-5000]
        if period_changed:
            self._fouls = {"home": None, "away": None}

        body: list[Event] = []
        for side in SIDES:
            new = getattr(reading, side)
            if new is None:
                continue
            old = self._score.get(side)
            self._score[side] = new
            if old is not None and new != old:
                body += self._on_delta(side, new - old, ts)
        for side in SIDES:
            new = getattr(reading, f"fouls_{side}")
            if new is None:
                continue
            old = self._fouls.get(side)
            self._fouls[side] = new
            if old is not None and new > old:
                body.append(self._foul_candidate(side, new, new - old, ts))
        body += self._expire(ts)

        key = (s.home, s.away, s.period, s.fouls_home, s.fouls_away, s.running)
        clock_due = (s.clock != self._last_clock
                     and ts - self._last_emit_ts >= self.cfg.scoreboard_min_interval)
        if key != self._last_key or clock_due:
            self._last_key, self._last_emit_ts, self._last_clock = key, ts, s.clock
            out.append(s.replace(conf=dict(s.conf)).to_event(self.game_id))
        self._prune(ts)
        return out + body

    def flush(self, now_ts: float) -> list[Event]:
        """Läuft Fenster ab: nicht bestätigte optische Treffer werden zu Fehlwürfen.
        Nur wenn die Tafel frisch ist (letzte Messung höchstens ``board_stale_s`` her)."""
        if self._last_ts is None or now_ts - self._last_ts > self.cfg.board_stale_s:
            return []
        return self._expire(now_ts)

    # ------------------------------------------------------------ Punkte
    def _candidates(self, side: Side, part: int, ts: float, used: set[str]):
        lo, hi = ts - self.cfg.window_before, ts + self.cfg.window_after
        want = EventType.FREE_THROW if part == 1 else EventType.SHOT
        best = None
        for r in self._recs.values():
            e = r.ev
            if r.consumed or e.id in used or e.type != want or e.team != side or not _made(e):
                continue
            if not lo <= e.video_ts <= hi:
                continue
            exact = part == 1 or e.payload.get("value") == part
            key = (not exact, abs(e.video_ts - ts))
            if best is None or key < best[0]:
                best = (key, r, exact)
        return None if best is None else (best[1], best[2])

    @staticmethod
    def _decompositions(d: int) -> list[list[int]]:
        res: list[list[int]] = []

        def rec(rem: int, mx: int, cur: list[int]) -> None:
            if rem == 0:
                res.append(list(cur))
                return
            if len(cur) >= 6:
                return
            for p in (3, 2, 1):
                if p <= mx and p <= rem:
                    cur.append(p)
                    rec(rem - p, p, cur)
                    cur.pop()

        rec(d, 3, [])
        return res

    def _plan(self, side: Side, d: int, ts: float) -> list[int]:
        best, best_score = [d], -1e9
        for parts in self._decompositions(d):
            used: set[str] = set()
            score = 0.0
            for p in parts:
                m = self._candidates(side, p, ts, used)
                if m is None:
                    score -= 0.7 if p == 1 else 0.6
                else:
                    used.add(m[0].ev.id)
                    score += 1.0 if m[1] else 0.5
            if score > best_score + 1e-9:
                best, best_score = parts, score
        return best

    def _on_delta(self, side: Side, d: int, ts: float) -> list[Event]:
        if d < 0:
            return [self._flag("score_decrease", f"Tafel {side}: {d} Punkte (Korrektur?)", side, ts, delta=d)]
        if d > self.cfg.max_jump:
            return [self._flag("score_jump", f"Tafel {side}: +{d} nicht auflösbar", side, ts, delta=d)]
        parts = self._plan(side, d, ts)
        out: list[Event] = []
        if len(parts) > 1 or d > 3:
            out.append(self._flag("multi_point_jump",
                                  f"Tafel {side}: +{d} aufgeteilt in {parts}", side, ts,
                                  delta=d, parts=parts))
        used: set[str] = set()
        for p in parts:
            m = self._candidates(side, p, ts, used)
            if m is not None:
                rec = m[0]
                used.add(rec.ev.id)
                ev = self._confirm(rec, p, ts)
                if ev is not None:
                    out.append(ev)
            else:
                out.append(self._backfill(side, p, ts))
        return out

    def _replace(self, rec: _Rec, new: Event) -> None:
        self._recs.pop(rec.ev.id, None)
        rec.ev = new
        self._recs[new.id] = rec

    def _confirm(self, rec: _Rec, part: int, ts: float) -> Optional[Event]:
        ev = rec.ev
        rec.consumed = True
        if ev.source == EventSource.CONSOLE and ev.status == EventStatus.CONFIRMED:
            return None  # Mensch hat das letzte Wort
        ch: dict = dict(status=EventStatus.CONFIRMED, source=EventSource.SYSTEM,
                        confidence=max(ev.confidence, self.cfg.confirmed_conf))
        if part in (2, 3) and ev.payload.get("value") != part:
            ch["payload"] = {"value": part, "three_prob": 1.0 if part == 3 else 0.0,
                             "board_corrected": True}
        if ev.period == 0 or ev.clock is None:
            ch.update(self._stamp(ev.video_ts))
        new = ev.derive(**ch)
        self._replace(rec, new)
        return new

    def _backfill(self, side: Side, part: int, ts: float) -> Event:
        want = EventType.FREE_THROW if part == 1 else EventType.SHOT
        best = None
        for r in self._recs.values():
            e = r.ev
            if r.consumed or e.type != want or e.team != side or e.payload.get("made", True):
                continue
            if e.source == EventSource.CONSOLE and e.status == EventStatus.CONFIRMED:
                continue
            if not ts - self.cfg.miss_search_before <= e.video_ts <= ts + self.cfg.window_after:
                continue
            key = (part > 1 and e.payload.get("value") != part, abs(e.video_ts - ts))
            if best is None or key < best[0]:
                best = (key, r)
        if best is not None:
            rec = best[1]
            ev = rec.ev
            payload: dict = {"made": True, "board_inferred": True}
            if part > 1:
                payload.update(value=part, three_prob=1.0 if part == 3 else 0.0)
            ch: dict = dict(payload=payload, source=EventSource.SYSTEM,
                            status=EventStatus.PROVISIONAL, confidence=self.cfg.missed_make_conf)
            if ev.period == 0 or ev.clock is None:
                ch.update(self._stamp(ev.video_ts))
            new = ev.derive(**ch)
            rec.consumed = True
            self._replace(rec, new)
            return new
        vts = ts - self.cfg.default_lag
        if part == 1:
            payload = {"made": True, "attempt": 1, "of": 1, "attempt_unknown": True}
            typ = EventType.FREE_THROW
        else:
            payload = {"value": part, "made": True, "x": 0.0, "y": 0.0,
                       "three_prob": 1.0 if part == 3 else 0.0, "location_unknown": True}
            typ = EventType.SHOT
        new = Event(game_id=self.game_id, type=typ, video_ts=vts, team=side,
                    payload=payload, confidence=self.cfg.new_shot_conf,
                    source=EventSource.SYSTEM, status=EventStatus.PROVISIONAL,
                    **self._stamp(vts))
        self._recs[new.id] = _Rec(new, consumed=True)
        self._resolved.append({"kind": "created", "side": side, "part": part, "ts": ts, "id": new.id})
        return new

    def _expire(self, horizon: float) -> list[Event]:
        out = []
        for rec in list(self._recs.values()):
            e = rec.ev
            if rec.consumed or e.type not in (EventType.SHOT, EventType.FREE_THROW) or not _made(e):
                continue
            if e.source != EventSource.OPTICAL or e.status != EventStatus.PROVISIONAL:
                continue
            if e.video_ts + self.cfg.window_before < horizon:
                new = e.derive(payload={"made": False, "expired_no_board_change": True},
                               source=EventSource.SYSTEM, status=EventStatus.PROVISIONAL,
                               confidence=self.cfg.expired_make_conf)
                rec.consumed = True
                self._replace(rec, new)
                out.append(new)
        return out

    # ------------------------------------------------------------ Fouls, Flags
    def _foul_candidate(self, side: Side, total: int, delta: int, ts: float) -> Event:
        ref = None
        best = None
        for r in self._recs.values():
            e = r.ev
            if e.type != EventType.FOUL or e.team != side:
                continue
            if e.payload.get("kind") == FoulKind.TECHNICAL.value or e.payload.get("kind") == FoulKind.TECHNICAL:
                continue
            dt = abs(e.video_ts - ts)
            if dt <= self.cfg.window_before + self.cfg.window_after and (best is None or dt < best):
                best, ref = dt, e.id
        return self._flag("foul_candidate",
                          f"Teamfouls {side} auf {total} gestiegen (+{delta}); Foul aus Konsole/Bogen "
                          f"prüfen, technische Fouls zählen nicht als Teamfoul",
                          side, ts, ref=ref, team_fouls=total, delta=delta)

    def _flag(self, kind: str, message: str, side: Optional[Side], ts: float,
              ref: Optional[str] = None, **extra) -> Event:
        return Event(game_id=self.game_id, type=EventType.FLAG, video_ts=ts, team=side,
                     payload={"kind": kind, "message": message, "ref": ref, **extra},
                     confidence=1.0, source=EventSource.SYSTEM, status=EventStatus.PROVISIONAL,
                     **self._stamp(ts))

    def reconcile_with_scoresheet(self, final_board: Union[ScoreboardReading, tuple, None],
                                  scoresheet_home: int, scoresheet_away: int) -> list[Event]:
        """Endstand Tafel vs. Spielbericht. Abweichung -> FLAG "scoreboard_mismatch"
        (der Bericht gewinnt)."""
        if final_board is None:
            bh, ba = self._state.home, self._state.away
        elif isinstance(final_board, ScoreboardReading):
            bh, ba = final_board.home, final_board.away
        else:
            bh, ba = final_board
        if (bh, ba) == (scoresheet_home, scoresheet_away):
            return []
        ts = self._last_ts or 0.0
        ev = self._flag("scoreboard_mismatch",
                        f"Tafel {bh}:{ba} weicht vom Spielbericht {scoresheet_home}:{scoresheet_away} ab "
                        f"(Bericht gilt)", None, ts,
                        board_home=bh, board_away=ba,
                        sheet_home=scoresheet_home, sheet_away=scoresheet_away, winner="scoresheet")
        return [ev]

    def _prune(self, ts: float) -> None:
        ttl = self.cfg.record_ttl
        for k in [k for k, r in self._recs.items() if r.ev.video_ts < ts - ttl and r.consumed]:
            del self._recs[k]
        self._resolved = [r for r in self._resolved if r["ts"] > ts - ttl]
