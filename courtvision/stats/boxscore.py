"""Box-Score als reine Projektion des Event-Logs (Event Sourcing).

``compute_boxscore`` ruft zuerst :func:`effective_events` auf; ersetzte und
VOID-Events zählen also nie. Alles ist deterministisch aus den Events ableitbar.

Konventionen
------------
* Zeit: Zwischen zwei aufeinanderfolgenden Events mit ``period > 0`` und
  ``clock`` wird die Spielzeitdifferenz (:func:`game_seconds`) den fünf Spielern
  auf dem Feld gutgeschrieben. Periodenende (``PERIOD_END``/``GAME_END``) zählt
  bis zur Restzeit 0 der Periode; beim Wechsel in die nächste Periode ohne
  PERIOD_END wird die vorige Periode ebenfalls bis zum Ende aufgefüllt.
  Events ohne Zeit (period 0 / clock None) zählen für Statistiken, nicht für
  die Zeitmessung (``PERIOD_END``/``GAME_END`` ohne Uhr gelten als Restzeit 0).
* Plus/Minus, Lineup-Punkte: Punkte werden der Besetzung gutgeschrieben, die im
  Moment des Scoring-Events (in Event-Reihenfolge) auf dem Feld steht. Das gilt
  auch für Freiwürfe nach einem Wechsel bei gleicher Uhrzeit (FIBA-Konvention
  vereinfacht: wer beim Freiwurf-Event auf dem Feld ist, bekommt den Punkt).
* Teamfouls (``team_fouls_by_period``) zählen alle Fouls außer technischen;
  technische Fouls zählen nur zu PF des Spielers.
* Spielstand: SCORESHEET (offizieller Bericht) > letztes SCOREBOARD > Summe der
  Punkte. ``score_mismatch``: mit Bericht -> Anzeigetafel (falls vorhanden, sonst
  Optik) weicht vom Bericht ab; ohne Bericht -> Anzeigetafel weicht von der
  optischen Summe ab.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..domain.events import (
    STAT_EVENT_TYPES,
    Event,
    EventStatus,
    EventType,
    FoulKind,
    effective_events,
    game_seconds,
)
from ..domain.models import SIDES, Game, Player, Side, other_side


class Counting(BaseModel):
    """Zählstatistiken (Spieler und Team teilen dieselben Felder)."""

    pts: int = 0
    fgm: int = 0
    fga: int = 0
    fg2m: int = 0
    fg2a: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    orb: int = 0
    drb: int = 0
    reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    blka: int = 0  # eigene Würfe, die geblockt wurden
    tov: int = 0
    pf: int = 0
    fd: int = 0  # gezogene Fouls

    @property
    def efg_pct(self) -> Optional[float]:
        return (self.fgm + 0.5 * self.fg3m) / self.fga if self.fga else None

    @property
    def ts_pct(self) -> Optional[float]:
        den = 2.0 * (self.fga + 0.44 * self.fta)
        return self.pts / den if den else None

    def _shot(self, value: int, made: bool) -> None:
        self.fga += 1
        self.fgm += int(made)
        if value == 3:
            self.fg3a += 1
            self.fg3m += int(made)
        else:
            self.fg2a += 1
            self.fg2m += int(made)
        if made:
            self.pts += value

    def _ft(self, made: bool) -> None:
        self.fta += 1
        if made:
            self.ftm += 1
            self.pts += 1

    def _reb(self, offensive: bool) -> None:
        self.reb += 1
        if offensive:
            self.orb += 1
        else:
            self.drb += 1


class PlayerLine(Counting):
    player_id: str
    team: Side
    number: int = 0
    name: str = ""
    seconds_played: float = 0.0
    plus_minus: int = 0
    passes: int = 0
    possession_s: float = 0.0
    shots: list[dict[str, Any]] = Field(default_factory=list)
    provisional_events: int = 0

    @property
    def minutes(self) -> float:
        return self.seconds_played / 60.0


class TeamLine(BaseModel):
    side: Side
    totals: Counting = Field(default_factory=Counting)
    team_rebounds: int = 0
    team_turnovers: int = 0
    points: int = 0
    team_fouls_by_period: dict[int, int] = Field(default_factory=dict)


class LineupLine(BaseModel):
    side: Side
    players: tuple[str, ...]
    seconds: float = 0.0
    pts_for: int = 0
    pts_against: int = 0

    @property
    def net(self) -> int:
        return self.pts_for - self.pts_against


class BoxScore(BaseModel):
    game_id: str
    home: TeamLine
    away: TeamLine
    players: dict[str, PlayerLine] = Field(default_factory=dict)
    lineups: list[LineupLine] = Field(default_factory=list)
    score: dict[str, int] = Field(default_factory=lambda: {"home": 0, "away": 0})
    score_mismatch: bool = False
    status: Literal["provisional", "final"] = "provisional"

    def team(self, side: Side) -> TeamLine:
        return self.home if side == "home" else self.away


_END_TYPES = (EventType.PERIOD_END, EventType.GAME_END)


def _with_offensive_foul_turnovers(evs: list[Event], window_s: float = 3.0) -> list[Event]:
    """Offensivfoul = Turnover (Ereignisregeln). Fouls kommen von Konsole/Bogen;
    fehlt ein passender TURNOVER(kind=offensive_foul) desselben Spielers im
    Zeitfenster, wird er für die Projektion ergänzt (nicht ins Log geschrieben)."""
    tos = [e for e in evs if e.type == EventType.TURNOVER and e.payload.get("kind") == "offensive_foul"]
    extra = []
    for f in evs:
        if f.type != EventType.FOUL or f.payload.get("kind") != "offensive" or not f.team:
            continue
        if any(t.player_id == f.player_id and abs(t.video_ts - f.video_ts) <= window_s for t in tos):
            continue
        extra.append(Event(game_id=f.game_id, id=f"{f.id}-to", type=EventType.TURNOVER, video_ts=f.video_ts,
                           seq=f.seq, period=f.period, clock=f.clock, team=f.team, player_id=f.player_id,
                           payload={"kind": "offensive_foul", "derived_from": f.id},
                           confidence=f.confidence, source=f.source, status=f.status))
    if not extra:
        return evs
    out = evs + extra
    out.sort(key=lambda e: (e.video_ts, e.seq))
    return out


def compute_boxscore(
    events: list[Event],
    game: Game,
    rosters: dict[str, list[Player]],
    provisional_threshold: float = 0.8,
) -> BoxScore:
    """Berechnet den Box-Score aus den Events (siehe Modul-Docstring)."""
    evs = _with_offensive_foul_turnovers(effective_events(events))
    teams: dict[Side, TeamLine] = {s: TeamLine(side=s) for s in SIDES}
    lines: dict[str, PlayerLine] = {}
    side_of: dict[str, Side] = {}
    for side in SIDES:
        for p in rosters.get(side, []):
            side_of[p.id] = side
            lines[p.id] = PlayerLine(player_id=p.id, team=side, number=p.number, name=p.name)

    def line_for(pid: str, side: Optional[Side]) -> PlayerLine:
        ln = lines.get(pid)
        if ln is None:
            s = side or side_of.get(pid) or "home"
            side_of[pid] = s
            ln = lines[pid] = PlayerLine(player_id=pid, team=s, name=pid)
        return ln

    on: dict[Side, set[str]] = {"home": set(), "away": set()}
    lineups: dict[tuple[Side, tuple[str, ...]], LineupLine] = {}

    def lineup(side: Side) -> Optional[LineupLine]:
        if len(on[side]) != 5:
            return None
        key = (side, tuple(sorted(on[side])))
        if key not in lineups:
            lineups[key] = LineupLine(side=side, players=key[1])
        return lineups[key]

    clock_state: dict[str, Any] = {"period": None, "t": 0.0}

    def gs(period: int, clock: Optional[float]) -> float:
        return game_seconds(period, clock, game.period_length_s, game.overtime_length_s, game.periods)

    def credit(until: float) -> None:
        dt = until - clock_state["t"]
        if dt <= 0:
            return
        clock_state["t"] = until
        for side in SIDES:
            for pid in on[side]:
                line_for(pid, side).seconds_played += dt
            lu = lineup(side)
            if lu is not None:
                lu.seconds += dt

    def advance(period: int, clock: Optional[float]) -> None:
        cur = clock_state["period"]
        if cur is None:
            clock_state["period"] = period
            clock_state["t"] = gs(period, None)
        elif period > cur:
            credit(gs(cur, 0.0))
            clock_state["period"] = period
            clock_state["t"] = gs(period, None)
        elif period < cur:
            return  # außer der Reihe: nicht für Zeitmessung verwenden
        credit(gs(period, clock))

    def score_points(side: Side, n: int) -> None:
        teams[side].points += n
        for pid in on[side]:
            line_for(pid, side).plus_minus += n
        for pid in on[other_side(side)]:
            line_for(pid, other_side(side)).plus_minus -= n
        mine, theirs = lineup(side), lineup(other_side(side))
        if mine is not None:
            mine.pts_for += n
        if theirs is not None:
            theirs.pts_against += n

    scoreboard: Optional[dict[str, int]] = None
    scoresheet: Optional[dict[str, int]] = None
    finalized = False
    any_provisional = False

    for ev in evs:
        # ---- Zeit
        if ev.type in _END_TYPES:
            p = ev.period if ev.period > 0 else clock_state["period"]
            if p:
                advance(p, ev.clock if ev.clock is not None else 0.0)
        elif ev.period > 0 and ev.clock is not None:
            advance(ev.period, ev.clock)

        t, pid = ev.type, ev.player_id
        side: Optional[Side] = ev.team or (side_of.get(pid) if pid else None)
        pl = line_for(pid, side) if pid else None
        if pl is not None and side is None:
            side = pl.team
        tl = teams[side] if side else None
        pay = ev.payload

        if t in STAT_EVENT_TYPES and ev.status == EventStatus.PROVISIONAL:
            any_provisional = True
            if ev.confidence < provisional_threshold:
                involved = {pid} | {
                    pay.get(k) for k in ("shooter_id", "fouled_id", "victim_id", "player_in", "player_out")
                }
                for i in involved:
                    if i and i in lines:
                        lines[i].provisional_events += 1

        if t == EventType.STARTERS and side:
            on[side] = set(pay.get("players", []))
            for i in on[side]:
                line_for(i, side)
        elif t == EventType.SUBSTITUTION and side:
            on[side].discard(pay.get("player_out"))
            if pay.get("player_in"):
                on[side].add(pay["player_in"])
                line_for(pay["player_in"], side)
        elif t == EventType.SHOT and side:
            value, made = int(pay.get("value", 2)), bool(pay.get("made"))
            for c in (tl.totals, pl):
                if c is not None:
                    c._shot(value, made)
            if pl is not None:
                pl.shots.append({"x": pay.get("x"), "y": pay.get("y"), "value": value,
                                 "made": made, "event_id": ev.id})
            if made:
                score_points(side, value)
        elif t == EventType.FREE_THROW and side:
            made = bool(pay.get("made"))
            for c in (tl.totals, pl):
                if c is not None:
                    c._ft(made)
            if made:
                score_points(side, 1)
        elif t == EventType.REBOUND and side:
            off = bool(pay.get("offensive"))
            tl.totals._reb(off)
            if pl is not None:
                pl._reb(off)
            else:
                tl.team_rebounds += 1
        elif t == EventType.ASSIST and side:
            tl.totals.ast += 1
            if pl is not None:
                pl.ast += 1
        elif t == EventType.STEAL and side:
            tl.totals.stl += 1
            if pl is not None:
                pl.stl += 1
        elif t == EventType.BLOCK and side:
            tl.totals.blk += 1
            if pl is not None:
                pl.blk += 1
            sid = pay.get("shooter_id")
            if sid:
                shooter = line_for(sid, other_side(side))
                shooter.blka += 1
                teams[shooter.team].totals.blka += 1
        elif t == EventType.TURNOVER and side:
            tl.totals.tov += 1
            if pl is not None:
                pl.tov += 1
            else:
                tl.team_turnovers += 1
        elif t == EventType.FOUL and side:
            tl.totals.pf += 1
            if pl is not None:
                pl.pf += 1
            if pay.get("kind") != FoulKind.TECHNICAL.value and pay.get("kind") != FoulKind.TECHNICAL:
                tl.team_fouls_by_period[ev.period] = tl.team_fouls_by_period.get(ev.period, 0) + 1
            fid = pay.get("fouled_id")
            if fid:
                fouled = line_for(fid, other_side(side))
                fouled.fd += 1
                teams[fouled.team].totals.fd += 1
        elif t == EventType.PASS and pl is not None:
            pl.passes += 1
        elif t == EventType.POSSESSION and pl is not None:
            pl.possession_s += max(0.0, float(pay.get("end_ts", 0)) - float(pay.get("start_ts", 0)))
        elif t == EventType.SCOREBOARD:
            # Unvollständige Tafel-Lesungen (home/away None) ignorieren
            if pay.get("home") is not None and pay.get("away") is not None:
                scoreboard = {"home": int(pay["home"]), "away": int(pay["away"])}
        elif t == EventType.SCORESHEET:
            if pay.get("home") is not None and pay.get("away") is not None:
                scoresheet = {"home": int(pay["home"]), "away": int(pay["away"])}
        elif t == EventType.GAME_FINALIZED:
            finalized = True

    optical = {"home": teams["home"].points, "away": teams["away"].points}
    if scoresheet is not None:
        score = scoresheet
        mismatch = (scoreboard if scoreboard is not None else optical) != scoresheet
    elif scoreboard is not None:
        score = scoreboard
        mismatch = scoreboard != optical
    else:
        score, mismatch = optical, False

    return BoxScore(
        game_id=game.id,
        home=teams["home"],
        away=teams["away"],
        players=lines,
        lineups=sorted(lineups.values(), key=lambda lu: (lu.side, -lu.seconds, lu.players)),
        score=dict(score),
        score_mismatch=mismatch,
        status="final" if finalized or not any_provisional else "provisional",
    )
