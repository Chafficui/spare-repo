"""Live-Session eines Spiels und Cloud-Nachverarbeitung.

Verdrahtung (Plan: Live-Pfad / Fusion und Korrektur)::

    Beobachtungen -> EventEngine --optische Events--> ScoreboardFusion --\\
    Tafel-Messwerte -> PlausibilityFilter -> Fusion / PeriodTracker -------+--> EventStore.append

Jedes entstehende Event wird sofort in den Store geschrieben (Store-Listener
treiben die WebSocket-Live-Updates). Optische Events ohne Viertel/Uhr werden
ueber ``fusion.clock_at`` gestempelt.

Entscheidungen
--------------
* Die ``ClockState``-Beobachtungen gehen an Engine und PeriodTracker, die
  (gefilterten) Tafel-Messwerte an Fusion und PeriodTracker (Doppelspeisung ist
  idempotent). Die Engine bekommt keine Uhr aus den Messwerten, die
  CV-Pipeline liefert ``ClockState`` selbst.
* ``set_starters`` ist eine Eingabe des Bedieners: das STARTERS-Event wird als
  source CONSOLE / CONFIRMED abgelegt (und bleibt bei ``reprocess`` erhalten).
* Der Plausibilitaetsfilter entprellt per Default mit ``n_stable=2``. Bei 1 Hz Tafelrate
  (Simulator) gehen mit 3 Lesungen aufeinanderfolgende Freiwurf-Punkte (alle ~3 s) als
  Sprung +2 durch und werden von der Fusion falsch zugeordnet (Punkte-Genauigkeit sinkt
  messbar); bei hoeherer OCR-Rate kann ``plausibility=PlausibilityFilter(n_stable=3)`` gesetzt werden.
* ``calibration`` darf eine ``geometry.court.Calibration`` (mit ``zones``), ein
  Zonen-Dict (``substitution``/``bench_home``/``bench_away``) oder
  ``SubstitutionZones`` sein; daraus entstehen die Wechselzonen der Engine.

Cloud-Nachverarbeitung (:func:`reprocess`) -- Ersetzungsrichtlinie
-------------------------------------------------------------------
Der Cloud-Lauf (volle Aufloesung) erzeugt eine frische Eventliste. Danach gilt:

* BEHALTEN: alle gueltigen Events mit source CONSOLE oder SCORESHEET (Menschen
  haben das letzte Wort), GAME_FINALIZED, sowie Tafel-Events (source SCOREBOARD:
  SCOREBOARD, Perioden-/Spielgrenzen; Cloud-Duplikate werden nicht uebernommen).
* VOID (``void(..., SYSTEM)``): alle uebrigen gueltigen Live-Events (source
  OPTICAL oder SYSTEM, auch durch die Tafel bestaetigte: die Fusion bestaetigt
  im neuen Lauf erneut).
* NEU anhaengen: die gueltigen Events des Cloud-Laufs, ausser sie decken sich mit
  einem menschlich behandelten Ereignis (gleicher Typ + Team innerhalb von
  ``match_tol_s`` zu einem erhaltenen Menschen-Event oder zu einem Vorfahren
  seiner Ersatzkette, auch wenn der Mensch es per VOID entfernt hat). So kehrt
  ein von Menschen entferntes Fehl-Event nicht zurueck und eine Korrektur wird
  nicht dupliziert. ASSIST/BLOCK, deren Wurf verworfen wurde, werden auf das
  Menschen-Event umgehaengt bzw. verworfen.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Union

from courtvision.domain.events import (Event, EventSource, EventStatus, EventType,
                                       effective_events, void)
from courtvision.domain.models import Game, GameStatus, Player
from courtvision.domain.observations import ClockState, Observation, obs_ts
from courtvision.scoreboard.fusion import FusionConfig, ScoreboardFusion
from courtvision.scoreboard.plausibility import PlausibilityFilter
from courtvision.scoreboard.state import ScoreboardReading
from courtvision.scoring.engine import EngineConfig, EventEngine
from courtvision.scoring.substitutions import SubstitutionZones
from courtvision.store.eventstore import EventStore

from .period_tracker import PeriodTracker

Item = Union[Observation, ScoreboardReading]


def as_reading(x: Any) -> ScoreboardReading:
    """ScoreboardReading oder Sim-Format ``(video_ts, {feld: wert})``."""
    if isinstance(x, ScoreboardReading):
        return x
    ts, d = x
    return ScoreboardReading(video_ts=ts, **d)


def merge_streams(observations: Iterable[Observation], readings: Iterable[Any]) -> Iterator[Item]:
    """Mischt zwei nach ``video_ts`` sortierte Strome; bei gleicher Zeit zuerst die Tafel."""
    a = ((as_reading(r).video_ts, 0, i, as_reading(r)) for i, r in enumerate(readings))
    b = ((obs_ts(o), 1, i, o) for i, o in enumerate(observations))
    for _, _, _, item in heapq.merge(a, b, key=lambda t: (t[0], t[1])):
        yield item


def zones_from_calibration(calibration: Any) -> Optional[SubstitutionZones]:
    if calibration is None:
        return None
    if isinstance(calibration, SubstitutionZones):
        return calibration
    z = calibration if isinstance(calibration, dict) else getattr(calibration, "zones", None)
    if not z or "substitution" not in z:
        return None
    benches = {s: z[f"bench_{s}"] for s in ("home", "away") if f"bench_{s}" in z}
    default = SubstitutionZones()
    return SubstitutionZones(substitution_zone=list(z["substitution"]),
                             benches=benches or default.benches)


class GameSession:
    def __init__(self, store: EventStore, game: Game, rosters: dict[str, list[Player]], *,
                 engine_config: Optional[EngineConfig] = None,
                 fusion_config: Optional[FusionConfig] = None,
                 calibration: Any = None, use_filter: bool = True,
                 plausibility: Optional[PlausibilityFilter] = None,
                 end_settle_s: float = 4.0) -> None:
        self.store, self.game, self.rosters = store, game, rosters
        self.calibration = calibration
        self.engine = EventEngine(game.id, rosters, engine_config, zones_from_calibration(calibration))
        self.fusion = ScoreboardFusion(game.id, fusion_config)
        self.filter: Optional[PlausibilityFilter] = (
            (plausibility or PlausibilityFilter(n_stable=2)) if use_filter else None)
        self.tracker = PeriodTracker(game.id, game, end_settle_s=end_settle_s)
        self.last_ts = 0.0
        self.finished = False

    # ------------------------------------------------------------ Hilfen
    def _append(self, evs: Iterable[Event]) -> list[Event]:
        return [self.store.append(e) for e in evs]

    def _stamp(self, ev: Event) -> Event:
        if ev.period and ev.clock is not None:
            return ev
        pc = self.fusion.clock_at(ev.video_ts)
        if pc is None:
            return ev
        return ev.model_copy(update={"period": pc[0], "clock": pc[1]})

    def _optical(self, evs: Iterable[Event]) -> list[Event]:
        out: list[Event] = []
        for e in evs:
            e = self._stamp(e) if e.source == EventSource.OPTICAL else e
            out += self._append([e])
            out += self._append(self.fusion.on_optical_event(e))
        return out

    # ------------------------------------------------------------ Eingang
    def set_starters(self, side: str, ids: Iterable[str]) -> list[Event]:
        evs = [e.model_copy(update={"source": EventSource.CONSOLE, "status": EventStatus.CONFIRMED,
                                    "confidence": 1.0})
               for e in self.engine.set_starters(side, list(ids))]
        return self._append(evs)

    def feed_observation(self, obs: Observation) -> list[Event]:
        self.last_ts = max(self.last_ts, obs_ts(obs))
        out = self._optical(self.engine.feed(obs))
        if isinstance(obs, ClockState):
            out += self._append(self.tracker.feed(obs))
        return out

    def feed_scoreboard(self, reading: Any) -> list[Event]:
        r = as_reading(reading)
        self.last_ts = max(self.last_ts, r.video_ts)
        out: list[Event] = []
        if self.filter is not None:
            fo = self.filter.update(r)
            flags = [self._stamp(Event(
                game_id=self.game.id, type=EventType.FLAG, video_ts=a.video_ts,
                payload={"kind": f"scoreboard_{a.kind}", "message": a.message, "ref": None,
                         "field": a.field},
                source=EventSource.SYSTEM, status=EventStatus.PROVISIONAL))
                for a in fo.anomalies if a.accepted]
            out += self._append(flags)
            if fo.reading is None:
                return out
            r = fo.reading
        if r.home is not None and r.away is not None:
            # SCOREBOARD-Events brauchen beide Staende (Box-Score liest sie als int); der Filter
            # liefert waehrend der Anfangsentprellung nur Teilzustaende.
            out += self._append(self.fusion.on_reading(r))
        out += self._append(self.tracker.feed(r))
        return out

    def feed_many(self, items: Iterable[Any], readings: Optional[Iterable[Any]] = None) -> list[Event]:
        """Verarbeitet Beobachtungen und Tafel-Messwerte.

        * ``feed_many(observations, readings)``: beide (je nach ``video_ts`` sortiert)
          werden gemischt, bei gleicher Zeit zuerst die Tafel.
        * ``feed_many(merged)``: ein bereits gemischter Strom aus Observationen und
          ``ScoreboardReading`` (bzw. ``(ts, dict)``-Paaren), in gegebener Reihenfolge.
        """
        stream = merge_streams(items, readings) if readings is not None else items
        out: list[Event] = []
        for it in stream:
            if isinstance(it, (ScoreboardReading, tuple)):
                out += self.feed_scoreboard(it)
            else:
                out += self.feed_observation(it)
        return out

    def finish(self, video_ts: Optional[float] = None) -> list[Event]:
        """Abschluss: Engine/Fusion leeren, Perioden und Spiel beenden, Status PROCESSING."""
        if self.finished:
            return []
        self.finished = True
        ts = self.last_ts if video_ts is None else video_ts
        out = self._optical(self.engine.flush())
        out += self._append(self.fusion.flush(self.fusion._last_ts if self.fusion._last_ts is not None else ts))
        out += self._append(self.tracker.end(ts))
        try:
            self.game = self.store.set_game_status(self.game.id, GameStatus.PROCESSING)
        except KeyError:
            self.game = self.game.model_copy(update={"status": GameStatus.PROCESSING})
        return out

    def events(self) -> list[Event]:
        return self.store.effective(self.game.id)


# ---------------------------------------------------------------------------
# Cloud-Nachverarbeitung
# ---------------------------------------------------------------------------
_HUMAN = (EventSource.CONSOLE, EventSource.SCORESHEET)


@dataclass
class ReprocessReport:
    voided: int = 0
    added: int = 0
    kept_human: int = 0
    kept_scoreboard: int = 0
    suppressed: int = 0  # Cloud-Events, die ein Mensch bereits behandelt hat


def _is_kept(ev: Event) -> bool:
    return (ev.source in _HUMAN or ev.source == EventSource.SCOREBOARD
            or ev.type == EventType.GAME_FINALIZED)


def reprocess(store: EventStore, game: Game, rosters: dict[str, list[Player]],
              observations: Iterable[Observation], readings: Iterable[Any], *,
              engine_config: Optional[EngineConfig] = None,
              fusion_config: Optional[FusionConfig] = None,
              calibration: Any = None, match_tol_s: float = 2.0) -> ReprocessReport:
    """Ersetzt die vorlaeufigen optischen Live-Events durch das Ergebnis des Cloud-Laufs
    (Richtlinie siehe Modul-Docstring)."""
    rep = ReprocessReport()
    all_live = store.events(game.id)
    live = effective_events(all_live)
    by_id = {e.id: e for e in all_live}

    # frischer Lauf auf Scratch-Store (gleiche Aufstellung wie live)
    scratch = EventStore(":memory:")
    sess = GameSession(scratch, game, rosters, engine_config=engine_config,
                       fusion_config=fusion_config, calibration=calibration)
    starters = {e.team: e.payload.get("players") for e in live
                if e.type == EventType.STARTERS and e.team}
    for side, ids in starters.items():
        if ids:
            sess.set_starters(side, ids)
    sess.feed_many(observations, readings)
    sess.finish()
    fresh = scratch.effective(game.id)

    # menschlich behandelte Ereignisse (inkl. Ahnen der Ersatzkette) als Belegung
    human = [e for e in live if e.source in _HUMAN and e.type in (
        EventType.SHOT, EventType.FREE_THROW, EventType.REBOUND, EventType.ASSIST, EventType.STEAL,
        EventType.BLOCK, EventType.TURNOVER, EventType.FOUL, EventType.SUBSTITUTION)]
    human_ids = {e.id for e in human}
    claims: list[Event] = list(human)
    for e in all_live:
        if e.source in _HUMAN and e.supersedes:
            cur = by_id.get(e.supersedes)
            while cur is not None:
                if cur.source not in _HUMAN:
                    claims.append(cur)
                cur = by_id.get(cur.supersedes) if cur.supersedes else None

    live_has_board = any(e.source == EventSource.SCOREBOARD for e in live)
    live_period_keys = {(e.type, e.period) for e in live if e.source == EventSource.SCOREBOARD
                        and e.type != EventType.SCOREBOARD}

    def claimed_by(ev: Event) -> Optional[Event]:
        for c in claims:
            if c.type == ev.type and c.team == ev.team and abs(c.video_ts - ev.video_ts) <= match_tol_s:
                return c
        return None

    voids: list[Event] = []
    for e in live:
        if _is_kept(e):
            if e.source in _HUMAN or e.type == EventType.GAME_FINALIZED:
                rep.kept_human += 1
            else:
                rep.kept_scoreboard += 1
        else:
            voids.append(void(e, EventSource.SYSTEM, "reprocess: Cloud-Lauf ersetzt Live-Event"))

    new: list[Event] = []
    remap: dict[str, Optional[str]] = {}
    for e in fresh:
        if e.source == EventSource.SCOREBOARD:
            if e.type == EventType.SCOREBOARD and not live_has_board:
                pass  # keine Tafelhistorie vorhanden: uebernehmen
            elif e.type != EventType.SCOREBOARD and (e.type, e.period) not in live_period_keys:
                pass
            else:
                continue
        elif e.type == EventType.STARTERS and e.team in starters:
            continue  # Live-STARTERS (Bediener) bleiben
        elif e.type in (EventType.SHOT, EventType.FREE_THROW, EventType.REBOUND, EventType.STEAL,
                        EventType.TURNOVER, EventType.FOUL, EventType.SUBSTITUTION, EventType.BLOCK,
                        EventType.ASSIST):
            c = claimed_by(e)
            if c is not None:
                rep.suppressed += 1
                remap[e.id] = c.id if c.id in human_ids and c.type == e.type else None
                continue
        new.append(e)

    dropped_shots = {k for k in remap}
    final_new: list[Event] = []
    for e in new:
        sid = e.payload.get("shot_id")
        if e.type in (EventType.ASSIST, EventType.BLOCK) and sid in dropped_shots:
            target = remap.get(sid)
            if target is None:
                rep.suppressed += 1
                continue
            e = e.model_copy(update={"payload": {**e.payload, "shot_id": target}})
        final_new.append(e.model_copy(update={"supersedes": None}))

    for v in voids:
        store.append(v)
    rep.voided = len(voids)
    for e in final_new:
        store.append(e)
    rep.added = len(final_new)
    return rep
