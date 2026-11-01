"""Human-in-the-Loop: Korrekturliste, Abgleich mit dem Spielberichtsbogen, Freigabe.

Alle Funktionen arbeiten auf *effektiven* Events (siehe ``domain.events``); jede
Korrektur ist ein neues Event (``confirm``/``correct``/``void``).
"""
from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from ..domain.events import (
    STAT_EVENT_TYPES,
    Event,
    EventSource,
    EventStatus,
    EventType,
    FoulKind,
    confirm,
    correct,
    effective_events,
    void,
)
from ..domain.models import GameStatus, Player, Side
from ..store.eventstore import EventStore

CLIP_BEFORE_S = 3.0
CLIP_AFTER_S = 2.0
MAX_SUGGESTIONS = 3
#: Mindestkonfidenz, unter der ein Stat-Event die Freigabe blockiert (haerter als die Queue-Schwelle).
HARD_FLOOR = 0.5
#: FLAG-Arten, die die Freigabe blockieren, solange sie nicht aufgeloest sind.
BLOCKING_FLAG_KINDS = frozenset({"scoresheet_mismatch"})


class FinalizationError(RuntimeError):
    """Freigabe verweigert: offene Review-Items unter der harten Untergrenze."""


class Suggestion(BaseModel):
    label: str
    action: Literal["correct", "void"] = "correct"
    #: Kwargs fuer ``correct()`` (``player_id``, ``payload`` (gemerged), ...).
    changes: dict[str, Any] = Field(default_factory=dict)


class ReviewItem(BaseModel):
    event: Event
    impact: float
    clip_start: float
    clip_end: float
    suggestions: list[Suggestion] = Field(default_factory=list)
    reason: str = "low_confidence"  # | "flag"


# --------------------------------------------------------------------- Impact
def impact_weight(ev: Event) -> float:
    """Gewicht eines Events fuer den Box-Score (vor Multiplikation mit 1-Konfidenz)."""
    t = ev.type
    p = ev.payload
    if t == EventType.SHOT:
        if p.get("made"):
            return 10.0 * int(p.get("value", 2))
        return 6.0
    if t == EventType.FREE_THROW:
        return 10.0 if p.get("made") else 3.0
    if t == EventType.SUBSTITUTION:
        return 8.0
    if t in (EventType.REBOUND, EventType.ASSIST, EventType.STEAL, EventType.BLOCK):
        return 5.0
    if t == EventType.TURNOVER:
        return 4.5
    if t == EventType.FOUL:
        return 2.5
    if t == EventType.FLAG:
        return 15.0
    return 1.0


def _impact(ev: Event) -> float:
    if ev.type == EventType.FLAG:
        return impact_weight(ev)
    return impact_weight(ev) * (1.0 - ev.confidence)


# ------------------------------------------------------------ on-court / roster
def _on_court_before(eff: Sequence[Event], idx: int, side: Optional[str]) -> list[str]:
    court: list[str] = []
    for e in eff[:idx]:
        if e.team != side:
            continue
        if e.type == EventType.STARTERS:
            court = list(e.payload.get("players", []))
        elif e.type == EventType.SUBSTITUTION:
            out_, in_ = e.payload.get("player_out"), e.payload.get("player_in")
            court = [p for p in court if p != out_]
            if in_ and in_ not in court:
                court.append(in_)
    return court


def _label(rosters: Mapping[str, Sequence[Player]], pid: str) -> str:
    for plist in rosters.values():
        for pl in plist:
            if pl.id == pid:
                return f"#{pl.number} {pl.name}"
    return f"Spieler {pid}"


def _alt_players(ev: Event, eff: Sequence[Event], idx: int,
                 rosters: Mapping[str, Sequence[Player]], exclude: Iterable[Optional[str]]) -> list[str]:
    ex = {x for x in exclude if x}
    out: list[str] = []
    for pid in _on_court_before(eff, idx, ev.team):
        if pid not in ex and pid not in out:
            out.append(pid)
    for pl in rosters.get(ev.team or "", []):
        if pl.id not in ex and pl.id not in out:
            out.append(pl.id)
    return out


def _suggest(ev: Event, eff: Sequence[Event], idx: int,
             rosters: Mapping[str, Sequence[Player]]) -> list[Suggestion]:
    p = ev.payload
    sugg: list[Suggestion] = []

    def alt_players(n: int) -> list[Suggestion]:
        return [
            Suggestion(label=_label(rosters, pid), changes={"player_id": pid})
            for pid in _alt_players(ev, eff, idx, rosters, [ev.player_id])[:n]
        ]

    t = ev.type
    if t == EventType.SHOT:
        val = int(p.get("value", 2))
        sugg.append(Suggestion(label=f"{5 - val}er statt {val}er", changes={"payload": {"value": 5 - val}}))
        made = bool(p.get("made"))
        sugg.append(Suggestion(label="verworfen" if made else "getroffen", changes={"payload": {"made": not made}}))
        sugg += alt_players(1)
    elif t == EventType.FREE_THROW:
        made = bool(p.get("made"))
        sugg.append(Suggestion(label="verworfen" if made else "getroffen", changes={"payload": {"made": not made}}))
        sugg += alt_players(2)
    elif t == EventType.REBOUND:
        off = bool(p.get("offensive"))
        sugg.append(Suggestion(label="Defensiv-Rebound" if off else "Offensiv-Rebound",
                               changes={"payload": {"offensive": not off}}))
        sugg += alt_players(2)
    elif t == EventType.SUBSTITUTION:
        cur = (p.get("player_in"), p.get("player_out"))
        for cand in p.get("candidates", []):
            if isinstance(cand, Mapping):
                pin = cand.get("player_in", p.get("player_in"))
                pout = cand.get("player_out", p.get("player_out"))
            else:
                pin, pout = cand, p.get("player_out")
            if (pin, pout) == cur:
                continue
            sugg.append(Suggestion(label=f"{_label(rosters, str(pin))} fuer {_label(rosters, str(pout))}",
                                   changes={"payload": {"player_in": pin, "player_out": pout}}))
    elif t == EventType.FLAG:
        sugg.append(Suggestion(label="Hinweis verwerfen", action="void"))
    else:  # ASSIST, STEAL, BLOCK, TURNOVER, FOUL
        sugg += alt_players(3)
    return sugg[:MAX_SUGGESTIONS]


# ----------------------------------------------------------------- Queue
def correction_queue(events: Iterable[Event], rosters: Mapping[str, Sequence[Player]],
                     threshold: float = 0.8) -> list[ReviewItem]:
    """Korrekturliste, absteigend nach Einfluss auf den Box-Score.

    ``rosters`` ist nach Seite (``"home"``/``"away"``) geschluesselt.
    """
    eff = effective_events(events)
    items: list[ReviewItem] = []
    for idx, ev in enumerate(eff):
        if ev.status != EventStatus.PROVISIONAL:
            continue
        if ev.type == EventType.FLAG:
            reason = "flag"
        elif ev.type in STAT_EVENT_TYPES and ev.confidence < threshold:
            reason = "low_confidence"
        else:
            continue
        items.append(ReviewItem(
            event=ev, impact=_impact(ev),
            clip_start=max(0.0, ev.video_ts - CLIP_BEFORE_S), clip_end=ev.video_ts + CLIP_AFTER_S,
            suggestions=_suggest(ev, eff, idx, rosters), reason=reason,
        ))
    items.sort(key=lambda i: (-i.impact, i.event.video_ts, i.event.seq))
    return items


Choice = Union[str, Suggestion]


def apply_review(item_event: Event, choice: Choice) -> Event:
    """``"confirm"`` bestaetigt, ``"reject"`` (oder ``"void"``) verwirft, ein
    :class:`Suggestion` wird als Korrektur angewandt. Liefert das neue Event
    (der Aufrufer haengt es an den Store an)."""
    if isinstance(choice, Suggestion):
        if choice.action == "void":
            return void(item_event, reason=choice.label)
        return correct(item_event, **_copy_changes(choice.changes))
    if choice == "confirm":
        return confirm(item_event)
    if choice in ("reject", "void"):
        return void(item_event, reason="rejected in review")
    raise ValueError(f"unknown review choice: {choice!r}")


def _copy_changes(ch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(ch)
    if "payload" in out and out["payload"] is not None:
        out["payload"] = dict(out["payload"])
    return out


# ----------------------------------------------------------- Scoresheet
def _tally(eff: Sequence[Event]) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, str]]:
    players: dict[str, dict[str, int]] = {}
    teams = {"home": 0, "away": 0}
    last_ref: dict[str, str] = {}

    def row(pid: str) -> dict[str, int]:
        return players.setdefault(pid, {"points": 0, "fouls": 0, "ftm": 0})

    for e in eff:
        pts = 0
        if e.type == EventType.SHOT and e.payload.get("made"):
            pts = int(e.payload.get("value", 2))
        elif e.type == EventType.FREE_THROW and e.payload.get("made"):
            pts = 1
        if pts and e.team in teams:
            teams[e.team] += pts
        if e.player_id:
            r = row(e.player_id)
            if pts:
                r["points"] += pts
                last_ref[f"{e.player_id}:points"] = e.id
            if e.type == EventType.FREE_THROW and e.payload.get("made"):
                r["ftm"] += 1
                last_ref[f"{e.player_id}:ftm"] = e.id
            if e.type == EventType.FOUL:
                r["fouls"] += 1
                last_ref[f"{e.player_id}:fouls"] = e.id
    return players, teams, last_ref


def reconcile_scoresheet(events: Iterable[Event], scoresheet: Union[Mapping[str, Any], Event]) -> list[Event]:
    """FLAG-Events (kind ``scoresheet_mismatch``) fuer Abweichungen zum Bogen.

    Verglichen werden Punkte, Fouls, getroffene Freiwuerfe je Spieler (und die
    Teampunkte, falls auf dem Bogen). Freiwurfversuche stehen nicht im Bogen und
    werden nicht verglichen.
    """
    sheet = scoresheet.payload if isinstance(scoresheet, Event) else scoresheet
    eff = [e for e in effective_events(events) if e.type != EventType.SCORESHEET]
    players, teams, last_ref = _tally(eff)
    game_id = eff[0].game_id if eff else str(sheet.get("game_id", ""))
    ts = eff[-1].video_ts if eff else 0.0
    flags: list[Event] = []

    def flag(msg: str, ref: Optional[str], **extra: Any) -> None:
        flags.append(Event(
            game_id=game_id, type=EventType.FLAG, video_ts=ts, source=EventSource.SYSTEM,
            status=EventStatus.PROVISIONAL, confidence=1.0, supersedes=None,
            payload={"kind": "scoresheet_mismatch", "message": msg, "ref": ref, **extra},
        ))

    sheet_players: Mapping[str, Mapping[str, int]] = sheet.get("players", {})
    for pid in sorted(set(sheet_players) | set(players)):
        auto = players.get(pid, {"points": 0, "fouls": 0, "ftm": 0})
        ref_row = sheet_players.get(pid, {})
        for metric in ("points", "fouls", "ftm"):
            a, s = auto[metric], int(ref_row.get(metric, 0))
            if a != s:
                flag(f"{pid}: {metric} auto={a} Bogen={s}", last_ref.get(f"{pid}:{metric}"),
                     player_id=pid, metric=metric, auto=a, sheet=s)
    for side in ("home", "away"):
        if side in sheet and int(sheet[side]) != teams[side]:
            flag(f"Team {side}: Punkte auto={teams[side]} Bogen={sheet[side]}", None,
                 team=side, metric="team_points", auto=teams[side], sheet=int(sheet[side]))
    return flags


# -------------------------------------------------------------- finalize
def open_review_items(events: Iterable[Event], floor: float = HARD_FLOOR) -> list[ReviewItem]:
    items = correction_queue(events, {}, threshold=floor)
    return [i for i in items
            if i.reason == "low_confidence" or i.event.payload.get("kind") in BLOCKING_FLAG_KINDS]


def finalize(store: EventStore, game_id: str, by: str, force: bool = False,
             hard_floor: float = HARD_FLOOR) -> Event:
    """Gibt das Spiel frei (GAME_FINALIZED, Status FINAL).

    Verweigert (``FinalizationError``), solange Stat-Events mit Konfidenz unter
    ``hard_floor`` oder unaufgeloeste Bogen-Abweichungen offen sind, ausser ``force=True``.
    """
    eff = store.effective(game_id)
    open_items = open_review_items(eff, hard_floor)
    if open_items and not force:
        raise FinalizationError(f"{len(open_items)} offene Review-Items unter Konfidenz {hard_floor}")
    ev = Event(
        game_id=game_id, type=EventType.GAME_FINALIZED,
        video_ts=eff[-1].video_ts if eff else 0.0,
        source=EventSource.CONSOLE, status=EventStatus.CONFIRMED,
        payload={"by": by, "forced": bool(open_items), "open_items": len(open_items)},
    )
    ev = store.append(ev)
    if store.get_game(game_id) is not None:
        store.set_game_status(game_id, GameStatus.FINAL)
    return ev


# ----------------------------------------------------------- Console input
class ConsoleFoulInput:
    """Ein-Klick-Foul-Erfassung der Kampfgericht-Konsole."""

    def __init__(self, game_id: str, store: Optional[EventStore] = None) -> None:
        self.game_id = game_id
        self.store = store

    def foul(self, player_id: str, team: Side, kind: FoulKind = FoulKind.PERSONAL,
             fouled_id: Optional[str] = None, video_ts: float = 0.0, period: int = 0,
             clock: Optional[float] = None) -> Event:
        ev = Event(
            game_id=self.game_id, type=EventType.FOUL, video_ts=video_ts, period=period, clock=clock,
            team=team, player_id=player_id, source=EventSource.CONSOLE, status=EventStatus.CONFIRMED,
            confidence=1.0, payload={"kind": FoulKind(kind), "fouled_id": fouled_id},
        )
        return self.store.append(ev) if self.store is not None else ev


# ------------------------------------------------------------- Metriken
Box = Mapping[str, Mapping[str, float]]


def accuracy_report(auto_box: Box, reference_box: Box) -> dict[str, Any]:
    """Anteil exakt richtiger Werte und mittlerer absoluter Fehler.

    Eingabe: ``{player_id: {kategorie: wert}}``. Fehlende Werte zaehlen als 0.
    Ergebnis: ``per_category``, ``per_player``, ``overall`` mit ``exact_share``, ``mae``, ``n``.
    """
    pids = sorted(set(auto_box) | set(reference_box))
    cats = sorted({c for b in (auto_box, reference_box) for r in b.values() for c in r})
    cell: dict[tuple[str, str], tuple[bool, float]] = {}
    for pid in pids:
        for c in cats:
            a = float(auto_box.get(pid, {}).get(c, 0))
            r = float(reference_box.get(pid, {}).get(c, 0))
            cell[(pid, c)] = (a == r, abs(a - r))

    def agg(keys: Iterable[tuple[str, str]]) -> dict[str, float]:
        ks = list(keys)
        n = len(ks)
        if n == 0:
            return {"exact_share": 1.0, "mae": 0.0, "n": 0}
        return {"exact_share": sum(cell[k][0] for k in ks) / n, "mae": sum(cell[k][1] for k in ks) / n, "n": n}

    return {
        "per_category": {c: agg((p, c) for p in pids) for c in cats},
        "per_player": {p: agg((p, c) for c in cats) for p in pids},
        "overall": agg(cell.keys()),
    }


def event_f1(predicted: Iterable[Event], truth: Iterable[Event], tol_s: float = 1.0,
             match_player: bool = False) -> dict[str, dict[str, float]]:
    """Precision/Recall/F1 je Ereignistyp (plus ``"ALL"`` als Mikro-Mittel).

    Zuordnung optimal (Hungarian) nach Typ und |dt| <= ``tol_s`` (optional gleicher Spieler).
    """
    pred, tru = list(predicted), list(truth)
    types = sorted({e.type.value for e in pred} | {e.type.value for e in tru})
    out: dict[str, dict[str, float]] = {}
    tot_tp = tot_fp = tot_fn = 0
    for t in types:
        P = [e for e in pred if e.type.value == t]
        T = [e for e in tru if e.type.value == t]
        tp = 0
        if P and T:
            big = 1e9
            cost = np.full((len(P), len(T)), big)
            for i, p in enumerate(P):
                for j, q in enumerate(T):
                    dt = abs(p.video_ts - q.video_ts)
                    if dt <= tol_s and (not match_player or p.player_id == q.player_id):
                        cost[i, j] = dt
            r, c = linear_sum_assignment(cost)
            tp = int(sum(cost[i, j] < big for i, j in zip(r, c)))
        fp, fn = len(P) - tp, len(T) - tp
        out[t] = _prf(tp, fp, fn)
        tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn
    out["ALL"] = _prf(tot_tp, tot_fp, tot_fn)
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
