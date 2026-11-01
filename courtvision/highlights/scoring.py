"""Highlight-Score je Ereignis aus Typ, Spielsituation, Seltenheit und Konfidenz."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from ..domain.events import Event, EventStatus, EventType, FoulKind, effective_events
from ..domain.models import Game

CONF_THRESHOLD = 0.8
INJURY_MARGIN_S = 20.0
CLOSE_DIFF = 5
CLUTCH_SECONDS = 120.0
RUN_MIN_POINTS = 8
FASTBREAK_WINDOW_S = 8.0
AND_ONE_BEFORE_S = 1.0
AND_ONE_AFTER_S = 4.0
AND_ONE_FT_WINDOW_S = 15.0

#: Basis-Gewichte
W_THREE, W_TWO, W_BLOCK, W_STEAL, W_ASSIST, W_FT = 6.0, 3.0, 5.0, 2.0, 1.2, 0.5
B_FASTBREAK_SHOT, B_FASTBREAK_STEAL, B_AND_ONE, B_AND_ONE_FT = 2.0, 4.0, 3.0, 1.5

#: (player_id -> {kategorie: Anzahl bisher in der Saison})
History = Mapping[str, Mapping[str, int]]


@dataclass
class HighlightContext:
    score_diff: int = 99  # |Differenz| vor dem Ereignis
    score_after: tuple[int, int] = (0, 0)
    period: int = 1
    clock: Optional[float] = None
    close_game: bool = False
    clutch_time: bool = False
    run_points: int = 0  # unbeantwortete Punkte des Teams inkl. dieses Ereignisses
    fastbreak: bool = False
    and_one: bool = False
    prior_count: Optional[int] = None  # bisherige Saisonzahl dieser Art (None = unbekannt)
    injury: bool = False


@dataclass
class ScoredEvent:
    event: Event
    score: float
    context: HighlightContext = field(default_factory=HighlightContext)


def event_key(ev: Event) -> Optional[str]:
    """Kategorie fuer Seltenheit (Saison-Historie)."""
    if ev.type == EventType.SHOT and ev.payload.get("made"):
        return "three_made" if int(ev.payload.get("value", 2)) == 3 else "two_made"
    if ev.type == EventType.BLOCK:
        return "block"
    if ev.type == EventType.STEAL:
        return "steal"
    return None


def rarity_bonus(prior_count: Optional[int]) -> float:
    if prior_count is None:
        return 0.0
    if prior_count == 0:
        return 0.5
    if prior_count < 3:
        return 0.25
    return 0.0


# ------------------------------------------------------------- Szenenerkennung
def injury_windows(events: Iterable[Event], margin_s: float = INJURY_MARGIN_S) -> list[tuple[float, float]]:
    return [(e.video_ts - margin_s, e.video_ts + margin_s) for e in effective_events(events)
            if e.type == EventType.FLAG and e.payload.get("kind") == "injury"]


def in_injury_window(ts: float, windows: Iterable[tuple[float, float]]) -> bool:
    return any(a <= ts <= b for a, b in windows)


def detect_and_ones(eff: list[Event]) -> dict[str, dict]:
    """shot_id -> {"foul": id, "fts": [ids]} fuer Treffer mit gezogenem Foul (kein Offensivfoul)."""
    out: dict[str, dict] = {}
    for s in eff:
        if s.type != EventType.SHOT or not s.payload.get("made") or not s.player_id:
            continue
        foul = next((f for f in eff if f.type == EventType.FOUL
                     and f.payload.get("fouled_id") == s.player_id
                     and f.payload.get("kind") not in (FoulKind.OFFENSIVE, FoulKind.OFFENSIVE.value)
                     and -AND_ONE_BEFORE_S <= f.video_ts - s.video_ts <= AND_ONE_AFTER_S), None)
        if foul is None:
            continue
        fts = [f.id for f in eff if f.type == EventType.FREE_THROW and f.player_id == s.player_id
               and 0 <= f.video_ts - s.video_ts <= AND_ONE_FT_WINDOW_S]
        out[s.id] = {"foul": foul.id, "fts": fts}
    return out


def detect_fastbreaks(eff: list[Event], window_s: float = FASTBREAK_WINDOW_S) -> dict[str, str]:
    """shot_id -> steal_id: Steal gefolgt von einem Treffer desselben Teams innerhalb des Fensters."""
    out: dict[str, str] = {}
    for st in eff:
        if st.type != EventType.STEAL:
            continue
        for e in eff:
            if e.video_ts <= st.video_ts:
                continue
            if e.video_ts - st.video_ts > window_s:
                break
            if e.type == EventType.SHOT:
                if e.team == st.team and e.payload.get("made"):
                    out[e.id] = st.id
                break  # erster Wurf entscheidet
    return out


def build_contexts(events: Iterable[Event], game: Optional[Game] = None,
                   history: Optional[History] = None) -> dict[str, HighlightContext]:
    """Kontext je Event-id aus dem Event-Log (Spielstand aus SHOT/FREE_THROW und SCOREBOARD)."""
    eff = effective_events(events)
    periods = game.periods if game else 4
    windows = injury_windows(eff)
    and_ones = detect_and_ones(eff)
    and_one_ids = set(and_ones)
    for v in and_ones.values():
        and_one_ids.update(v["fts"])
    fast = detect_fastbreaks(eff)
    fast_ids = set(fast) | set(fast.values())
    history = history or {}

    home = away = 0
    run_team: Optional[str] = None
    run_pts = 0
    ctxs: dict[str, HighlightContext] = {}
    for e in eff:
        if e.type == EventType.SCOREBOARD:
            home, away = int(e.payload.get("home", home)), int(e.payload.get("away", away))
        before = abs(home - away)
        pts = 0
        if e.type == EventType.SHOT and e.payload.get("made"):
            pts = int(e.payload.get("value", 2))
        elif e.type == EventType.FREE_THROW and e.payload.get("made"):
            pts = 1
        if pts and e.team:
            if e.team == "home":
                home += pts
            else:
                away += pts
            if run_team == e.team:
                run_pts += pts
            else:
                run_team, run_pts = e.team, pts
        key = event_key(e)
        prior = None
        if key and e.player_id and e.player_id in history:
            prior = int(history[e.player_id].get(key, 0))
        clock = e.clock
        ctxs[e.id] = HighlightContext(
            score_diff=before, score_after=(home, away), period=e.period, clock=clock,
            close_game=before <= CLOSE_DIFF,
            clutch_time=e.period >= periods and clock is not None and clock <= CLUTCH_SECONDS,
            run_points=run_pts if pts and e.team == run_team else 0,
            fastbreak=e.id in fast_ids, and_one=e.id in and_one_ids,
            prior_count=prior, injury=in_injury_window(e.video_ts, windows),
        )
    return ctxs


# ------------------------------------------------------------------- Score
def highlight_score(event: Event, context: Optional[HighlightContext] = None) -> float:
    """Highlight-Score >= 0; 0 = kein Highlight (Typ ohne Gewicht, Verletzungsszene)."""
    c = context or HighlightContext()
    if c.injury:
        return 0.0
    t, p = event.type, event.payload
    base = 0.0
    if t == EventType.SHOT and p.get("made"):
        base = W_THREE if int(p.get("value", 2)) == 3 else W_TWO
        if c.fastbreak:
            base += B_FASTBREAK_SHOT
        if c.and_one:
            base += B_AND_ONE
    elif t == EventType.BLOCK:
        base = W_BLOCK
    elif t == EventType.STEAL:
        base = W_STEAL + (B_FASTBREAK_STEAL if c.fastbreak else 0.0)
    elif t == EventType.ASSIST:
        base = W_ASSIST
    elif t == EventType.FREE_THROW and p.get("made"):
        base = W_FT + (B_AND_ONE_FT if c.and_one else 0.0)
    if base <= 0:
        return 0.0
    mult = 1.0
    if c.close_game:
        mult += 0.6
    if c.clutch_time:
        mult += 1.0
    if c.run_points >= RUN_MIN_POINTS:
        mult += 0.6
    mult *= 1.0 + rarity_bonus(c.prior_count)
    return base * mult * max(0.0, min(1.0, event.confidence))


def publishable_confidence(event: Event, threshold: float = CONF_THRESHOLD) -> bool:
    """Automatisch veroeffentlichbar: ueber der Schwelle oder von einem Menschen bestaetigt."""
    return event.status == EventStatus.CONFIRMED or event.confidence >= threshold


def score_events(events: Iterable[Event], game: Optional[Game] = None, history: Optional[History] = None,
                 threshold: float = CONF_THRESHOLD) -> list[ScoredEvent]:
    """Bewertete Kandidaten (absteigend); ohne Verletzungsszenen und unsichere Events."""
    eff = effective_events(events)
    ctxs = build_contexts(eff, game, history)
    out = []
    for e in eff:
        c = ctxs[e.id]
        if c.injury or not publishable_confidence(e, threshold):
            continue
        s = highlight_score(e, c)
        if s > 0:
            out.append(ScoredEvent(e, s, c))
    out.sort(key=lambda x: (-x.score, x.event.video_ts))
    return out
