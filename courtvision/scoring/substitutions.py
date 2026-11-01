"""Spielerwechselerkennung (Produktplan, Abschnitt "Spielerwechselerkennung").

Ablauf
------
* Wechsel werden nur bei *gestoppter* Uhr gesucht (``ClockState.running`` False
  oder von aussen gesetzte Stillstandsgruende wie Freiwurf/Auszeit via
  :meth:`SubstitutionDetector.set_stoppage`).
* Basislinie ``L[side]``: die letzten fuenf Tracks eines Teams im Feld, solange
  die Uhr laeuft. Waehrend des Stillstands ist sie eingefroren.
* Beim Wiederanpfiff (Stillstand -> Uhr laeuft) wird pro Team verglichen:
  ausgetretene Tracks (in ``L``, jetzt ausserhalb, Richtung Bank/Zone) und
  eingetretene Tracks (nicht in ``L``, jetzt im Feld, vorher in Wechselzone/Bank
  bzw. am Spielfeldrand gesehen) werden paarweise zu offenen Wechseln.
  Aenderungen bei laufender Uhr erzeugen nie einen Wechsel.
* Identitaet: ``player_id`` des Tracks sobald gelesen; bis dahin Kandidatenliste =
  Kader minus aktuelle Fuenf minus Gesperrte. Bleibt genau ein Kandidat
  uebrig, wird per Ausschluss aufgeloest (niedrigere Konfidenz).
* Plausibilitaet: ausgehender Spieler muss auf dem Feld gewesen sein; ein
  einlaufender, bereits auf dem Feld stehender oder gesperrter Spieler wird
  verworfen (FLAG-Event) und bleibt unaufgeloest.
* Fallback: nach 30 s *Spielzeit* (nur Zeit bei laufender Uhr) ohne Aufloesung
  wird ein SUBSTITUTION mit niedriger Konfidenz und ``candidates`` emittiert
  (Korrekturliste). Wird der Track spaeter doch gelesen, folgt ein ersetzendes
  Event (``Event.derive``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..domain.events import Event, EventSource, EventStatus, EventType
from ..domain.models import Player, Side
from ..domain.observations import ClockState, FrameState, Observation
from ..geometry.court import in_court, point_in_polygon

Polygon = Sequence[tuple[float, float]]


@dataclass
class SubstitutionZones:
    """Wechselzone und Baenke als Polygone in Feldmetern (aus der Kalibrierung)."""

    substitution_zone: Polygon = field(
        default_factory=lambda: [(-2.5, 7.5), (2.5, 7.5), (2.5, 9.5), (-2.5, 9.5)]
    )
    benches: dict[str, Polygon] = field(
        default_factory=lambda: {
            "home": [(-12.0, 7.5), (-2.5, 7.5), (-2.5, 10.0), (-12.0, 10.0)],
            "away": [(2.5, 7.5), (12.0, 7.5), (12.0, 10.0), (2.5, 10.0)],
        }
    )

    def in_zone(self, x: float, y: float) -> bool:
        if point_in_polygon(x, y, self.substitution_zone):
            return True
        return any(point_in_polygon(x, y, p) for p in self.benches.values())


@dataclass
class SubstitutionConfig:
    unresolved_after_s: float = 30.0  # Spielzeit (Uhr laeuft) bis Fallback
    debounce_s: float = 0.5  # Track muss so lange im/ausser Feld sein
    stop_grace_s: float = 3.0  # Tafel-Latenz: Austritt kurz vor Stillstand zaehlt noch
    min_id_confidence: float = 0.5
    tentative_id_confidence: float = 0.2  # darunter: nicht mal als Tipp
    boundary_margin_m: float = 1.5  # "am Rand" als Austrittsbeleg
    require_zone_evidence: bool = True
    court_margin_m: float = 0.0
    team_size: int = 5
    resolved_confidence: float = 0.9
    elimination_confidence: float = 0.6
    late_confidence: float = 0.8
    fallback_confidence: float = 0.25


@dataclass
class _Pending:
    side: str
    in_track: int
    out_track: Optional[int]
    ts: float
    period: int
    clock: Optional[float]
    run_at: float


class SubstitutionDetector:
    def __init__(
        self,
        rosters: dict[str, list[Player]],
        zones: Optional[SubstitutionZones] = None,
        config: Optional[SubstitutionConfig] = None,
        game_id: str = "game",
    ) -> None:
        self.rosters = {s: list(ps) for s, ps in rosters.items()}
        self.zones = zones or SubstitutionZones()
        self.cfg = config or SubstitutionConfig()
        self.game_id = game_id
        self._on: dict[str, set[str]] = {s: set() for s in ("home", "away")}
        self._disq: set[str] = set()
        self._track_pid: dict[int, str] = {}
        self._tentative: dict[int, tuple[str, float]] = {}
        self._track_side: dict[int, str] = {}
        self._court: dict[int, tuple[bool, float]] = {}  # track -> (im Feld, seit)
        self._last_pos: dict[int, tuple[float, float]] = {}
        self._zone_seen: dict[int, float] = {}
        self._edge_seen: dict[int, bool] = {}  # erste Sichtung am Rand
        self._base: dict[str, set[int]] = {"home": set(), "away": set()}
        self._reasons: set[str] = set()
        self._clock: Optional[ClockState] = None
        self._ts = 0.0
        self._run_time = 0.0
        self._stop_start: Optional[float] = None
        self._was_stopped = False
        self._pending: list[_Pending] = []
        self._fallbacks: list[tuple[Event, _Pending]] = []
        self._flagged: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------ API
    def set_starters(self, side: str, player_ids: Iterable[str]) -> list[Event]:
        ids = list(player_ids)
        if len(ids) != self.cfg.team_size or len(set(ids)) != len(ids):
            raise ValueError(f"need {self.cfg.team_size} distinct starters, got {ids}")
        self._on[side] = set(ids)
        return [self._event(EventType.STARTERS, self._ts, side, None, None, {"players": ids}, 1.0)]

    def on_court(self, side: str) -> set[str]:
        return set(self._on[side])

    def disqualify(self, player_id: str) -> None:
        self._disq.add(player_id)

    def set_stoppage(self, reason: str, active: bool) -> list[Event]:
        """Externer Stillstand (z. B. "free_throw", "timeout") an/aus."""
        was = self._stopped()
        (self._reasons.add if active else self._reasons.discard)(reason)
        return self._after_state_change(was, self._ts)

    def feed(self, obs: Observation) -> list[Event]:
        out: list[Event] = []
        ts = obs.video_ts
        if self._clock is not None and self._clock.running is True and ts > self._ts:
            self._run_time += ts - self._ts
        self._ts = max(self._ts, ts)
        was = self._stopped()
        if isinstance(obs, ClockState):
            self._clock = obs
            out += self._after_state_change(was, ts)
        elif isinstance(obs, FrameState):
            self._on_frame(obs)
        out += self._tick(ts)
        return out

    def flush(self) -> list[Event]:
        out: list[Event] = []
        if self._stopped():
            out += self._evaluate(self._ts)
        for p in list(self._pending):
            out += self._emit_fallback(p)
        self._pending.clear()
        return out

    # ------------------------------------------------------------ internals
    def _stopped(self) -> bool:
        return bool(self._reasons) or (self._clock is not None and self._clock.running is False)

    def _after_state_change(self, was: bool, ts: float) -> list[Event]:
        now = self._stopped()
        out: list[Event] = []
        if now and not was:
            self._stop_start = ts
        elif was and not now:
            out += self._evaluate(ts)
            self._stop_start = None
        return out

    def _roster_ids(self, side: str) -> list[str]:
        return [p.id for p in self.rosters.get(side, [])]

    def _on_frame(self, f: FrameState) -> None:
        ts = f.video_ts
        seen: set[int] = set()
        present: dict[str, set[int]] = {"home": set(), "away": set()}
        for ps in f.players:
            if ps.team is None:
                continue
            t = ps.track_id
            seen.add(t)
            self._track_side[t] = ps.team
            if ps.player_id is not None:
                if ps.id_confidence >= self.cfg.min_id_confidence:
                    self._track_pid[t] = ps.player_id
                elif ps.id_confidence >= self.cfg.tentative_id_confidence:
                    self._tentative[t] = (ps.player_id, ps.id_confidence)
            inc = in_court(ps.x, ps.y, self.cfg.court_margin_m)
            if t not in self._court:
                self._edge_seen[t] = self._near_edge(ps.x, ps.y)
            if t not in self._court or self._court[t][0] != inc:
                self._court[t] = (inc, ts)
            self._last_pos[t] = (ps.x, ps.y)
            if not inc and self.zones.in_zone(ps.x, ps.y):
                self._zone_seen[t] = ts
            if inc:
                present[ps.team].add(t)
        for t, (inc, _) in list(self._court.items()):
            if t not in seen and inc:  # verschwunden -> nicht mehr im Feld
                self._court[t] = (False, ts)
        if not self._stopped():
            for side in ("home", "away"):
                if len(present[side]) == self.cfg.team_size:
                    self._base[side] = set(present[side])
        else:
            for side in ("home", "away"):
                if not self._base[side] and len(present[side]) == self.cfg.team_size:
                    self._base[side] = set(present[side])

    def _near_edge(self, x: float, y: float) -> bool:
        m = self.cfg.boundary_margin_m
        return not in_court(x, y, -m)

    def _in_court(self, t: int) -> bool:
        return self._court.get(t, (False, 0.0))[0]

    def _since(self, t: int) -> float:
        return self._court.get(t, (False, 0.0))[1]

    def _evaluate(self, ts: float) -> list[Event]:
        """Wiederanpfiff: offene Wechsel je Team bilden."""
        out: list[Event] = []
        c = self.cfg
        start = (self._stop_start if self._stop_start is not None else ts) - c.stop_grace_s
        for side in ("home", "away"):
            base = self._base[side]
            present = {t for t, (inc, _) in self._court.items() if inc and self._track_side.get(t) == side}
            outs = sorted(
                (
                    t for t in base
                    if t not in present
                    and self._since(t) >= start
                    and ts - self._since(t) >= c.debounce_s
                    and self._exit_evidence(t)
                ),
                key=self._since,
            )
            ins = sorted(
                (
                    t for t in present
                    if t not in base
                    and self._since(t) >= start
                    and ts - self._since(t) >= c.debounce_s
                    and self._entry_evidence(t)
                ),
                key=self._since,
            )
            pairs: list[tuple[Optional[int], int]] = []
            for i, t_in in enumerate(ins):
                if i < len(outs):
                    pairs.append((outs[i], t_in))
                elif len(present) <= c.team_size:
                    pairs.append((None, t_in))
            for t_out, t_in in pairs:
                p = _Pending(
                    side, t_in, t_out, self._since(t_in),
                    self._clock.period if self._clock else 0,
                    self._clock.clock if self._clock else None,
                    self._run_time,
                )
                self._pending.append(p)
            if pairs:
                # neue Basislinie, damit der Wechsel nicht doppelt erkannt wird
                self._base[side] = (base - set(outs)) | set(ins)
        out += self._resolve_pending(ts, final=False)
        return out

    def _exit_evidence(self, t: int) -> bool:
        pos = self._last_pos.get(t)
        if pos is None:
            return True
        if self.zones.in_zone(*pos):
            return True
        return self._near_edge(*pos)

    def _entry_evidence(self, t: int) -> bool:
        if not self.cfg.require_zone_evidence:
            return True
        return t in self._zone_seen or self._edge_seen.get(t, False)

    # -------------------------------------------------------- Identitaet
    def _candidates_in(self, side: str) -> list[str]:
        return [i for i in self._roster_ids(side) if i not in self._on[side] and i not in self._disq]

    def _candidates_out(self, side: str, p: _Pending) -> list[str]:
        known = {
            self._track_pid[t] for t, (inc, _) in self._court.items()
            if inc and self._track_side.get(t) == side and t in self._track_pid and t != p.in_track
        }
        used = {self._track_pid.get(q.out_track) for q in self._pending if q is not p and q.out_track is not None}
        return [i for i in sorted(self._on[side]) if i not in known and i not in used]

    def _flag(self, p: _Pending, kind: str, message: str) -> Optional[Event]:
        key = (kind, p.in_track)
        if key in self._flagged:
            return None
        self._flagged.add(key)
        return self._event(EventType.FLAG, p.ts, p.side, None, p.in_track,
                           {"kind": kind, "message": message, "ref": None}, 0.5, p.period, p.clock)

    def _resolve(self, p: _Pending, out: list[Event]):
        side = p.side
        pin = self._track_pid.get(p.in_track)
        if pin is not None:
            if pin in self._disq:
                f = self._flag(p, "disqualified_player_entered", f"{pin} is disqualified")
                out += [f] if f else []
                pin = None
            elif pin in self._on[side] or pin not in self._roster_ids(side):
                f = self._flag(p, "implausible_substitution", f"{pin} cannot enter (on court or not in roster)")
                out += [f] if f else []
                pin = None
        pout = self._track_pid.get(p.out_track) if p.out_track is not None else None
        if pout is not None and pout not in self._on[side]:
            f = self._flag(p, "implausible_substitution", f"{pout} was not on court")
            out += [f] if f else []
            pout = None
        elim = False
        cand_in = self._candidates_in(side)
        cand_out = self._candidates_out(side, p)
        if pin is None and len(cand_in) == 1:
            pin, elim = cand_in[0], True
        if pout is None and len(cand_out) == 1:
            pout, elim = cand_out[0], True
        return pin, pout, elim, cand_in, cand_out

    def _resolve_pending(self, ts: float, final: bool) -> list[Event]:
        out: list[Event] = []
        for p in list(self._pending):
            pin, pout, elim, _, _ = self._resolve(p, out)
            if pin is not None and pout is not None:
                conf = self.cfg.elimination_confidence if elim else self.cfg.resolved_confidence
                self._on[p.side].discard(pout)
                self._on[p.side].add(pin)
                out.append(self._event(EventType.SUBSTITUTION, p.ts, p.side, pin, p.in_track,
                                       {"player_in": pin, "player_out": pout}, conf, p.period, p.clock))
                self._pending.remove(p)
            elif final or self._run_time - p.run_at >= self.cfg.unresolved_after_s:
                out += self._emit_fallback(p)
                self._pending.remove(p)
        return out

    def _emit_fallback(self, p: _Pending) -> list[Event]:
        out: list[Event] = []
        pin, pout, _, cand_in, cand_out = self._resolve(p, out)
        if pin is None:
            tent = self._tentative.get(p.in_track)
            pin = tent[0] if tent and tent[0] in cand_in else None
        if pout is None and p.out_track is not None:
            tent = self._tentative.get(p.out_track)
            pout = tent[0] if tent and tent[0] in cand_out else None
        payload = {
            "player_in": pin or "unknown",
            "player_out": pout or "unknown",
            "candidates": list(cand_in),
        }
        ev = self._event(EventType.SUBSTITUTION, p.ts, p.side, pin, p.in_track, payload,
                         self.cfg.fallback_confidence, p.period, p.clock)
        out.append(ev)
        if pin and pout:
            self._on[p.side].discard(pout)
            self._on[p.side].add(pin)
        if pin is None or pout is None:
            self._fallbacks.append((ev, p))
        return out

    def _tick(self, ts: float) -> list[Event]:
        out: list[Event] = []
        if self._pending:
            out += self._resolve_pending(ts, final=False)
        # spaete Identifikation nach Fallback -> ersetzendes Event
        for ev, p in list(self._fallbacks):
            pin = self._track_pid.get(p.in_track)
            pout = self._track_pid.get(p.out_track) if p.out_track is not None else None
            cur_in, cur_out = ev.payload["player_in"], ev.payload["player_out"]
            new_in = pin if pin and cur_in == "unknown" and pin not in self._on[p.side] and pin not in self._disq else cur_in
            new_out = pout if pout and cur_out == "unknown" else cur_out
            if (new_in, new_out) != (cur_in, cur_out):
                payload = {"player_in": new_in, "player_out": new_out}
                new = ev.derive(
                    player_id=None if new_in == "unknown" else new_in,
                    confidence=self.cfg.late_confidence if "unknown" not in (new_in, new_out) else ev.confidence,
                    payload={**payload, "candidates": [c for c in ev.payload.get("candidates", []) if c != new_in]},
                )
                out.append(new)
                self._fallbacks.remove((ev, p))
                if "unknown" in (new_in, new_out):
                    self._fallbacks.append((new, p))
                else:
                    self._on[p.side].discard(new_out)
                    self._on[p.side].add(new_in)
        return out

    def _event(self, typ, ts, side, pid, track, payload, conf, period=None, clock=None) -> Event:
        if period is None:
            period = self._clock.period if self._clock else 0
            clock = self._clock.clock if self._clock else None
        return Event(
            game_id=self.game_id, type=typ, video_ts=ts, period=period, clock=clock,
            team=side, player_id=pid, track_id=track, payload=payload, confidence=conf,
            source=EventSource.OPTICAL, status=EventStatus.PROVISIONAL,
        )
