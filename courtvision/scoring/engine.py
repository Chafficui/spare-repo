"""Regelbasierter Zustandsautomat: Observationen -> Events.

Eingabe: sortierter Strom von :mod:`courtvision.domain.observations`.
Ausgabe: :class:`courtvision.domain.events.Event` (source OPTICAL, PROVISIONAL).

Kernideen
---------
* Ballbesitz-Segmente (``Possession``) bilden die Zeitreihe; ``ShotRelease``,
  ``RimMake``, ``BallTouch``, ``OutOfBounds``, ``Formation`` liefern die
  Ereignisanker.
* Events werden emittiert, sobald sie "final genug" sind: Treffer sofort beim
  ``RimMake`` (inkl. Assist), Fehlwurf nach Ablauf des Trefferfensters bzw. beim
  naechsten Ballbesitz/Aus, Rebound beim ersten kontrollierten Besitz.
* ``SHOT``-Events tragen ``video_ts`` = Wurfzeitpunkt (nicht den Zeitpunkt der
  Emission).

Entscheidungen / Grenzen (siehe auch Tests)
-------------------------------------------
* Steal: Ballbesitzwechsel ohne Wurf mit ``BallTouch`` des Verteidigers im
  Fenster -> STEAL (+ TURNOVER steal). Ohne Touch: direkter Wechsel (Luecke
  <= ``direct_steal_gap_s``) -> STEAL mit niedriger Konfidenz, sonst
  TURNOVER ``bad_pass`` (niedrige Konfidenz).
* Sprungball, Freiwurfaufstellung, Aus, Uhr-Stopp, lange Luecken (> ``turnover_max_gap_s``)
  unterbrechen die Kette: dort entstehen weder Pass noch Turnover.
* Verstoesse, die optisch nicht sichtbar sind (24-s-Verstoss, Schrittfehler,
  Offensivfoul, Fouls allgemein), erzeugt die Engine NICHT; sie kommen ueber
  die Konsole bzw. den Spielberichtsbogen (Shot-Clock-Violation = Team-Turnover
  ``shot_clock`` per Konsole).
* Freiwurf: ``of`` ist optisch unbekannt (Default ``ft_default_of``, wird auf
  ``attempt`` angehoben). Letzter Versuch = Fehlwurf, nach dem ein anderer
  Spieler als der Werfer den Ball kontrolliert; ein weiterer Freiwurf-Release
  verwirft den Rebound-Kandidaten. Die Serie endet bei Feldwurf, laufender Uhr,
  Rebound oder ``ft_series_timeout_s``.
* Spielerwechsel: :class:`SubstitutionDetector` wird komponiert, wenn Kader
  uebergeben werden. Kein Quellbezug (Tafel) hier: ``optical_score`` ist rein
  optisch und wird nicht durch Korrekturen aktualisiert.
* ``track_id`` wird immer gesetzt (wenn bekannt), ``player_id`` nur wenn die
  Nummer bekannt ist; ``to_id`` eines PASS ist ohne ID ``"track-<id>"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..domain.events import Event, EventSource, EventStatus, EventType, TurnoverKind
from ..domain.models import Player, other_side
from ..domain.observations import (
    BallTouch,
    ClockState,
    Formation,
    FrameState,
    Observation,
    OutOfBounds,
    Possession,
    RimMake,
    ShotRelease,
)
from .substitutions import SubstitutionConfig, SubstitutionDetector, SubstitutionZones


@dataclass
class EngineConfig:
    make_window_s: float = 3.0  # Wurf -> RimMake
    min_flight_s: float = 0.35  # Besitz des Werfers direkt nach Release ignorieren
    block_window_s: float = 0.8  # Wurf -> abrupte Beruehrung des Gegners
    assist_window_s: float = 2.0  # Annahme -> Wurf-Release des Empfaengers
    rebound_window_s: float = 8.0  # Fehlwurf -> erster Besitz
    team_rebound_window_s: float = 20.0  # Aus -> folgender Besitz
    pass_max_gap_s: float = 2.5
    steal_touch_lookback_s: float = 1.0
    direct_steal_gap_s: float = 0.3
    turnover_max_gap_s: float = 4.0
    oob_lookback_s: float = 3.0
    min_possession_s: float = 0.3
    ft_default_of: int = 2
    ft_series_timeout_s: float = 25.0
    jump_ball_window_s: float = 10.0
    three_threshold: float = 0.5
    scramble_radius_m: float = 1.5
    scramble_max_players: int = 2  # mehr als so viele -> Gedraenge
    scramble_factor: float = 0.7
    unknown_id_factor: float = 0.8
    ambiguous_three_factor: float = 0.85  # three_prob in (0.25, 0.75)
    clip_post_s: float = 2.0
    clip_default_lead_s: float = 3.0  # Clipstart wenn Szenenbeginn unbekannt
    min_obs_confidence: float = 0.2  # Observationen darunter werden ignoriert
    steal_direct_confidence: float = 0.55
    bad_pass_confidence: float = 0.5
    emit_possessions: bool = True
    emit_passes: bool = True
    substitution: SubstitutionConfig = field(default_factory=SubstitutionConfig)


@dataclass
class _Seg:
    track: int
    team: str
    pid: Optional[str]
    start: float
    conf: float
    scene_start: float
    end: float = 0.0


@dataclass
class _Shot:
    ts: float
    track: int
    team: str
    pid: Optional[str]
    x: float
    y: float
    basket: str
    three_prob: float
    conf: float
    clip_start: float
    period: int
    clock: Optional[float]
    ft_attempt: int = 0
    assist: Optional[dict] = None


@dataclass
class _Rebound:
    team: str
    shot_id: str
    clip_start: float
    miss_ts: float
    ft: bool = False
    ft_shooter: Optional[int] = None
    pending_team: bool = False
    oob_ts: float = 0.0
    oob_team: Optional[str] = None
    period: int = 0
    clock: Optional[float] = None


class EventEngine:
    def __init__(
        self,
        game_id: str,
        rosters: Optional[dict[str, list[Player]]] = None,
        config: Optional[EngineConfig] = None,
        zones: Optional[SubstitutionZones] = None,
    ) -> None:
        self.game_id = game_id
        self.cfg = config or EngineConfig()
        self.rosters = rosters
        self.subs: Optional[SubstitutionDetector] = (
            SubstitutionDetector(rosters, zones, self.cfg.substitution, game_id) if rosters else None
        )
        self._ts = 0.0
        self._clock: Optional[ClockState] = None
        self._frame: Optional[FrameState] = None
        self._tracks: dict[int, str] = {}
        self._cur: Optional[_Seg] = None
        self._last: Optional[_Seg] = None
        self._dead = False
        self._scene_team: Optional[str] = None
        self._scene_start = 0.0
        self._touches: list[BallTouch] = []
        self._last_pass: Optional[dict[str, Any]] = None
        self._shot: Optional[_Shot] = None
        self._rb: Optional[_Rebound] = None
        self._ft: Optional[dict[str, Any]] = None
        self._jump: Optional[dict[str, Any]] = None
        self._score = {"home": 0, "away": 0}

    # ------------------------------------------------------------------ API
    @property
    def optical_score(self) -> dict[str, int]:
        return dict(self._score)

    def set_starters(self, side: str, player_ids: list[str]) -> list[Event]:
        if self.subs is None:
            raise RuntimeError("rosters required for starters")
        return self.subs.set_starters(side, player_ids)

    def on_court(self, side: str) -> set[str]:
        return self.subs.on_court(side) if self.subs else set()

    def disqualify(self, player_id: str) -> None:
        if self.subs:
            self.subs.disqualify(player_id)

    def feed(self, obs: Observation) -> list[Event]:
        out: list[Event] = []
        ts = obs.video_ts
        if self.subs:
            out += self.subs.feed(obs)
        self._ts = max(self._ts, ts)
        out += self._advance(ts)
        if getattr(obs, "confidence", 1.0) < self.cfg.min_obs_confidence:
            return out
        if isinstance(obs, ClockState):
            out += self._on_clock(obs)
        elif isinstance(obs, FrameState):
            self._on_frame(obs)
        elif isinstance(obs, Possession):
            out += self._on_possession(obs)
        elif isinstance(obs, ShotRelease):
            out += self._on_shot(obs)
        elif isinstance(obs, RimMake):
            out += self._on_rim(obs)
        elif isinstance(obs, BallTouch):
            out += self._on_touch(obs)
        elif isinstance(obs, OutOfBounds):
            out += self._on_oob(obs)
        elif isinstance(obs, Formation):
            out += self._on_formation(obs)
        return out

    def flush(self) -> list[Event]:
        out: list[Event] = []
        if self._shot:
            out += self._resolve_miss(min(self._ts, self._shot.ts + self.cfg.make_window_s))
        rb = self._rb
        if rb and rb.pending_team and rb.oob_team:
            out += self._team_rebound(rb, other_side(rb.oob_team), self._ts, 0.4)
        self._rb = None
        out += self._close_seg(self._ts)
        if self._jump:
            out += self._emit_jump(None)
        self._end_ft()
        if self.subs:
            out += self.subs.flush()
        return out

    # -------------------------------------------------------------- helpers
    def _ident(self, track: Optional[int], pid: Optional[str]) -> Optional[str]:
        if pid:
            return pid
        return self._tracks.get(track) if track is not None else None

    def _idf(self, pid: Optional[str]) -> float:
        return 1.0 if pid else self.cfg.unknown_id_factor

    def _mk(self, typ: EventType, ts: float, team: Optional[str], track: Optional[int],
            pid: Optional[str], payload: dict, conf: float, clip_start: Optional[float] = None,
            clip_end: Optional[float] = None, snap: Optional[tuple[int, Optional[float]]] = None) -> Event:
        if snap is None:
            snap = (self._clock.period, self._clock.clock) if self._clock else (0, None)
        return Event(
            game_id=self.game_id, type=typ, video_ts=ts, period=snap[0], clock=snap[1],
            team=team, player_id=pid, track_id=track, payload=payload,
            confidence=round(max(0.05, min(1.0, conf)), 4),
            source=EventSource.OPTICAL, status=EventStatus.PROVISIONAL,
            clip_start=clip_start, clip_end=clip_end,
        )

    def _post(self, ts: float) -> float:
        return ts + self.cfg.clip_post_s

    def _scramble(self) -> bool:
        f = self._frame
        if f is None or f.ball_x is None or f.ball_y is None:
            return False
        n = sum(
            1 for p in f.players
            if ((p.x - f.ball_x) ** 2 + (p.y - f.ball_y) ** 2) ** 0.5 <= self.cfg.scramble_radius_m
        )
        return n > self.cfg.scramble_max_players

    # ------------------------------------------------------------ time-based
    def _advance(self, ts: float) -> list[Event]:
        out: list[Event] = []
        c = self.cfg
        if self._shot and ts > self._shot.ts + c.make_window_s:
            out += self._resolve_miss(self._shot.ts + c.make_window_s)
        rb = self._rb
        if rb:
            if rb.pending_team:
                if ts - rb.oob_ts > c.team_rebound_window_s:
                    if rb.oob_team:
                        out += self._team_rebound(rb, other_side(rb.oob_team), rb.oob_ts, 0.4)
                    self._rb = None
            elif ts - rb.miss_ts > c.rebound_window_s:
                self._rb = None
        if self._ft and self._shot is None and ts - self._ft["last_ts"] > c.ft_series_timeout_s:
            out += self._end_ft()
        if self._jump and ts - self._jump["ts"] > c.jump_ball_window_s:
            out += self._emit_jump(None)
        self._touches = [t for t in self._touches if ts - t.video_ts <= 5.0]
        return out

    # ----------------------------------------------------------- observations
    def _on_clock(self, c: ClockState) -> list[Event]:
        out: list[Event] = []
        prev = self._clock
        self._clock = c
        if prev is not None and (prev.period != c.period or (prev.running and not c.running)):
            out += self._close_seg(c.video_ts)
            self._break_chain()
        if c.running and self._ft and self._shot is None:
            out += self._end_ft()
        return out

    def _on_frame(self, f: FrameState) -> None:
        self._frame = f
        for p in f.players:
            if p.player_id and p.id_confidence >= 0.5:
                self._tracks[p.track_id] = p.player_id

    def _break_chain(self) -> None:
        self._last = None
        self._last_pass = None
        self._touches.clear()

    def _close_seg(self, ts: float) -> list[Event]:
        seg = self._cur
        if seg is None:
            return []
        seg.end = ts
        self._cur = None
        self._last = seg
        if not self.cfg.emit_possessions or ts - seg.start < self.cfg.min_possession_s:
            return []
        return [self._mk(EventType.POSSESSION, seg.start, seg.team, seg.track, seg.pid,
                         {"start_ts": seg.start, "end_ts": ts}, seg.conf * self._idf(seg.pid),
                         clip_start=seg.start, clip_end=ts)]

    def _on_touch(self, t: BallTouch) -> list[Event]:
        s = self._shot
        if (s and t.abrupt_trajectory_change and t.team != s.team and s.ft_attempt == 0
                and t.video_ts - s.ts <= self.cfg.block_window_s):
            return self._block(t)
        self._touches.append(t)
        if t.player_id:
            self._tracks[t.track_id] = t.player_id
        return []

    def _on_possession(self, p: Possession) -> list[Event]:
        out: list[Event] = []
        ts = p.video_ts
        if p.track_id is None:
            return self._close_seg(ts)
        if p.team is None:
            return out  # Schiri / unklar
        if p.player_id:
            self._tracks[p.track_id] = p.player_id
        s = self._shot
        if s and p.track_id == s.track and ts - s.ts < self.cfg.min_flight_s:
            return out
        if s:
            out += self._resolve_miss(ts)
        if self._cur and self._cur.track == p.track_id:
            if p.player_id and not self._cur.pid:
                self._cur.pid = p.player_id
            return out
        pid = self._ident(p.track_id, p.player_id)
        prev_cur = self._cur
        out += self._close_seg(ts)
        last = self._last
        seg = _Seg(p.track_id, p.team, pid, ts, p.confidence, ts)
        # Freiwurf: Besitz des Werfers (Ball vom Schiri) aendert nichts
        if self._rb and self._rb.ft and p.track_id == self._rb.ft_shooter:
            self._cur, self._last = seg, None
            return out
        if self._jump:
            out += self._emit_jump(p.team)
            self._dead = True
        out += self._rebound(seg)
        if self._ft and self._rb is None and self._shot is None and self._ft.get("missed_last"):
            out += self._end_ft()
        if self._dead:
            last = None
            self._dead = False
        out += self._transition(last, seg) if last is not None else []
        if self._scene_team != seg.team:
            self._scene_team, self._scene_start = seg.team, seg.start
        seg.scene_start = self._scene_start
        self._cur = seg
        return out

    # ------------------------------------------------------ pass/steal/turnover
    def _transition(self, last: _Seg, seg: _Seg) -> list[Event]:
        c = self.cfg
        gap = seg.start - last.end
        out: list[Event] = []
        if last.team == seg.team:
            if last.track == seg.track or gap > c.pass_max_gap_s:
                return out
            to_id = seg.pid or f"track-{seg.track}"
            conf = min(last.conf, seg.conf) * self._idf(last.pid)
            if c.emit_passes:
                out.append(self._mk(EventType.PASS, last.end, last.team, last.track, last.pid,
                                    {"to_id": to_id}, conf, clip_start=last.scene_start,
                                    clip_end=self._post(seg.start)))
            self._last_pass = {"team": seg.team, "passer_track": last.track, "passer_pid": last.pid,
                               "recv_track": seg.track, "recv_pid": seg.pid, "recv_ts": seg.start,
                               "conf": conf}
            return out
        # Teamwechsel
        self._last_pass = None
        if gap > c.turnover_max_gap_s:
            return out
        touch = self._find_touch(last, seg)
        victim_scene = last.scene_start
        if touch is not None:
            self._touches.remove(touch)
            tpid = self._ident(touch.track_id, touch.player_id)
            conf = touch.confidence * seg.conf * self._idf(tpid) * (1.0 if tpid or touch.track_id == seg.track else 0.9)
            out.append(self._mk(EventType.STEAL, touch.video_ts, seg.team, touch.track_id, tpid,
                                {"victim_id": last.pid}, conf, clip_start=victim_scene,
                                clip_end=self._post(seg.start)))
            out.append(self._mk(EventType.TURNOVER, touch.video_ts, last.team, last.track, last.pid,
                                {"kind": TurnoverKind.STEAL.value}, conf * self._idf(last.pid),
                                clip_start=victim_scene, clip_end=self._post(seg.start)))
            self._scene_team, self._scene_start = seg.team, min(touch.video_ts, seg.start)
        elif gap <= c.direct_steal_gap_s:
            conf = c.steal_direct_confidence * seg.conf
            out.append(self._mk(EventType.STEAL, seg.start, seg.team, seg.track, seg.pid,
                                {"victim_id": last.pid}, conf * self._idf(seg.pid),
                                clip_start=victim_scene, clip_end=self._post(seg.start)))
            out.append(self._mk(EventType.TURNOVER, seg.start, last.team, last.track, last.pid,
                                {"kind": TurnoverKind.STEAL.value}, conf * self._idf(last.pid),
                                clip_start=victim_scene, clip_end=self._post(seg.start)))
            self._scene_team, self._scene_start = seg.team, seg.start
        else:
            out.append(self._mk(EventType.TURNOVER, last.end, last.team, last.track, last.pid,
                                {"kind": TurnoverKind.BAD_PASS.value},
                                c.bad_pass_confidence * last.conf * self._idf(last.pid),
                                clip_start=victim_scene, clip_end=self._post(seg.start)))
        return out

    def _find_touch(self, last: _Seg, seg: _Seg) -> Optional[BallTouch]:
        c = self.cfg
        cands = [
            t for t in self._touches
            if t.team == seg.team and last.end - c.steal_touch_lookback_s <= t.video_ts <= seg.start + 0.2
        ]
        if not cands:
            return None
        own = [t for t in cands if t.track_id == seg.track]
        return (own or cands)[-1]

    # --------------------------------------------------------------- shots
    def _on_shot(self, r: ShotRelease) -> list[Event]:
        out: list[Event] = []
        ts = r.video_ts
        if self._shot:
            out += self._resolve_miss(ts)
        ft = self._ft
        pid = self._ident(r.track_id, r.player_id)
        if self._rb:
            rb = self._rb
            if rb.ft:
                self._rb = None  # naechster Freiwurf: vorheriger war nicht der letzte
            elif not rb.pending_team and ft is None:
                # Putback ohne Besitzsignal: Werfer hat den Ball gesichert
                conf = 0.6 * r.confidence * self._idf(pid)
                out.append(self._mk(EventType.REBOUND, ts, r.team, r.track_id, pid,
                                    {"offensive": r.team == rb.team}, conf,
                                    clip_start=rb.clip_start, clip_end=self._post(ts),
                                    snap=(rb.period, rb.clock)))
                self._rb = None
            elif rb.pending_team:
                out += self._team_rebound(rb, r.team, ts, 0.5)
                self._rb = None
        out += self._close_seg(ts)
        scene_start = self._scene_start if self._scene_team == r.team else ts - self.cfg.clip_default_lead_s
        if ft is not None and ft["basket"] in (None, r.basket):
            ft["attempts"] += 1
            ft["last_ts"] = ts
            ft["missed_last"] = False
            attempt = ft["attempts"]
            clip_start = ft["ts"]
        else:
            if ft is not None:
                out += self._end_ft()
            attempt = 0
            clip_start = scene_start
            self._scene_team, self._scene_start = r.team, scene_start
        assist = None
        lp = self._last_pass
        if (attempt == 0 and lp and lp["team"] == r.team and lp["recv_track"] == r.track_id
                and lp["passer_track"] != r.track_id
                and 0 <= ts - lp["recv_ts"] <= self.cfg.assist_window_s):
            assist = lp
        self._last = None
        self._last_pass = None
        self._touches.clear()
        spid = pid or (ft["pid"] if attempt else None)
        self._shot = _Shot(ts, r.track_id, r.team, spid,
                           r.x, r.y, r.basket, r.three_prob, r.confidence, clip_start,
                           self._clock.period if self._clock else 0,
                           self._clock.clock if self._clock else None, attempt, assist)
        return out

    def _shot_event(self, s: _Shot, made: bool, done_ts: float, conf: float) -> Event:
        c = self.cfg
        snap = (s.period, s.clock)
        end = self._post(done_ts)
        if s.ft_attempt:
            return self._mk(EventType.FREE_THROW, s.ts, s.team, s.track, s.pid,
                            {"made": made, "attempt": s.ft_attempt,
                             "of": max(c.ft_default_of, s.ft_attempt)},
                            conf * self._idf(s.pid), s.clip_start, end, snap)
        value = 3 if s.three_prob >= c.three_threshold else 2
        if made and 0.25 < s.three_prob < 0.75:
            conf *= c.ambiguous_three_factor
        return self._mk(EventType.SHOT, s.ts, s.team, s.track, s.pid,
                        {"value": value, "made": made, "x": s.x, "y": s.y, "three_prob": s.three_prob},
                        conf * self._idf(s.pid), s.clip_start, end, snap)

    def _on_rim(self, m: RimMake) -> list[Event]:
        s = self._shot
        if s is None or m.basket != s.basket or m.video_ts - s.ts > self.cfg.make_window_s:
            return []
        out: list[Event] = []
        self._shot = None
        conf = min(s.conf, m.confidence)
        ev = self._shot_event(s, True, m.video_ts, conf)
        out.append(ev)
        if s.ft_attempt:
            self._score[s.team] += 1
            self._ft["last_ts"] = m.video_ts
            return out
        self._score[s.team] += ev.payload["value"]
        a = s.assist
        if a is not None:
            apid = self._ident(a["passer_track"], a["passer_pid"])
            out.append(self._mk(EventType.ASSIST, s.ts, s.team, a["passer_track"], apid,
                                {"shot_id": ev.id}, ev.confidence * min(1.0, a["conf"]) * 0.9,
                                s.clip_start, ev.clip_end, (s.period, s.clock)))
        self._rb = None
        return out

    def _resolve_miss(self, done_ts: float) -> list[Event]:
        s = self._shot
        if s is None:
            return []
        self._shot = None
        ev = self._shot_event(s, False, done_ts, s.conf)
        self._rb = _Rebound(s.team, ev.id, s.clip_start, done_ts, ft=bool(s.ft_attempt),
                            ft_shooter=s.track if s.ft_attempt else None,
                            period=s.period, clock=s.clock)
        if s.ft_attempt:
            self._ft["last_ts"] = done_ts
            self._ft["missed_last"] = True
        return [ev]

    def _block(self, t: BallTouch) -> list[Event]:
        s = self._shot
        self._shot = None
        out: list[Event] = []
        bpid = self._ident(t.track_id, t.player_id)
        shot_ev = self._shot_event(s, False, t.video_ts, s.conf)
        out.append(shot_ev)
        conf = min(s.conf, t.confidence) * self._idf(bpid)
        out.append(self._mk(EventType.BLOCK, t.video_ts, t.team, t.track_id, bpid,
                            {"shot_id": shot_ev.id, "shooter_id": s.pid}, conf,
                            s.clip_start, self._post(t.video_ts), (s.period, s.clock)))
        self._rb = _Rebound(s.team, shot_ev.id, s.clip_start, t.video_ts,
                            period=s.period, clock=s.clock)
        return out

    # ------------------------------------------------------------- rebounds
    def _rebound(self, seg: _Seg) -> list[Event]:
        rb = self._rb
        if rb is None:
            return []
        self._rb = None
        scramble = self._scramble()
        if rb.pending_team:
            return self._team_rebound(rb, seg.team, seg.start, 0.6 * seg.conf)
        conf = seg.conf * self._idf(seg.pid) * (self.cfg.scramble_factor if scramble else 1.0)
        if rb.ft:
            conf *= 0.95
            out = self._end_ft()
        else:
            out = []
        out.append(self._mk(EventType.REBOUND, seg.start, seg.team, seg.track, seg.pid,
                            {"offensive": seg.team == rb.team}, conf,
                            rb.clip_start, self._post(seg.start), (rb.period, rb.clock)))
        return out

    def _team_rebound(self, rb: _Rebound, team: str, ts: float, conf: float) -> list[Event]:
        return [self._mk(EventType.REBOUND, ts, team, None, None, {"offensive": team == rb.team},
                         conf, rb.clip_start, self._post(ts), (rb.period, rb.clock))]

    # ---------------------------------------------------------- out of bounds
    def _on_oob(self, o: OutOfBounds) -> list[Event]:
        out: list[Event] = []
        ts = o.video_ts
        if self._shot:
            out += self._resolve_miss(ts)
        src = self._cur or (self._last if self._last and ts - self._last.end <= self.cfg.oob_lookback_s else None)
        if self._rb:
            self._rb.pending_team = True
            self._rb.oob_ts = ts
            self._rb.oob_team = o.last_touch_team
            out += self._close_seg(ts)
            self._break_chain()
            self._dead = True
            return out
        if src is not None and (o.last_touch_team in (None, src.team)):
            pid = o.last_touch_player or self._ident(o.last_touch_track, None)
            track = o.last_touch_track
            if pid is None and track in (None, src.track):
                pid, track = src.pid, src.track
            conf = o.confidence * src.conf * self._idf(pid) * (1.0 if o.last_touch_team else 0.8)
            out.append(self._mk(EventType.TURNOVER, ts, src.team, track, pid,
                                {"kind": TurnoverKind.OUT_OF_BOUNDS.value}, conf,
                                src.scene_start, self._post(ts)))
        out += self._close_seg(ts)
        self._break_chain()
        self._dead = True
        return out

    # ------------------------------------------------------------ formations
    def _on_formation(self, f: Formation) -> list[Event]:
        out: list[Event] = []
        ts = f.video_ts
        if self._shot:
            out += self._resolve_miss(ts)
        out += self._close_seg(ts)
        self._break_chain()
        self._rb = None
        self._dead = True
        if f.kind == "jump_ball":
            self._jump = {"ts": ts, "conf": f.confidence, "snap": self._snap()}
            self._scene_team = None
        else:
            if self._ft:
                out += self._end_ft()
            self._ft = {"team": f.shooter_team, "shooter": f.shooter_track, "pid": f.shooter_player,
                        "basket": f.basket, "attempts": 0, "ts": ts, "last_ts": ts, "missed_last": False}
            self._scene_team, self._scene_start = f.shooter_team, ts
            if self.subs:
                out += self.subs.set_stoppage("free_throw", True)
        return out

    def _snap(self) -> tuple[int, Optional[float]]:
        return (self._clock.period, self._clock.clock) if self._clock else (0, None)

    def _emit_jump(self, winner: Optional[str]) -> list[Event]:
        j = self._jump
        self._jump = None
        return [self._mk(EventType.JUMP_BALL, j["ts"], winner, None, None, {"winner_team": winner},
                         j["conf"] * (1.0 if winner else 0.5), j["ts"], self._post(self._ts), j["snap"])]

    def _end_ft(self) -> list[Event]:
        if self._ft is None:
            return []
        self._ft = None
        return self.subs.set_stoppage("free_throw", False) if self.subs else []
