"""Human-in-the-Loop: Korrekturliste, Fouleingabe, Spielberichtsbogen, Freigabe."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from ..domain.events import Event, EventSource, EventStatus, EventType, effective_events, void
from ..domain.models import Game, GameStatus
from ..qa import (ConsoleFoulInput, FinalizationError, ReviewItem, apply_review, correction_queue, finalize,
                  reconcile_scoresheet)
from .auth import Principal, require
from .routes_live import FINAL_LOCK_MSG
from .schemas import FinalizeIn, FoulIn, ReviewIn, ScoresheetIn
from .state import AppState
from .text import Namer, describe, when, _resolver

REVIEW_ROLES = ("kampfgericht", "trainer")


def build(st: AppState) -> APIRouter:
    r = APIRouter()
    store = st.store

    def own_side(p: Principal, g: Game) -> Optional[str]:
        """Trainer duerfen nur eigene Mannschaft bearbeiten (None: keine Einschraenkung)."""
        if p.role != "trainer":
            return None
        try:
            return g.side_of(p.team_id or "")
        except KeyError:
            raise HTTPException(403, "Trainer gehört keinem der beiden Teams dieses Spiels an")

    def check_editable(g: Game, p: Principal) -> None:
        if g.status == GameStatus.FINAL and p.role != "admin":
            raise HTTPException(409, FINAL_LOCK_MSG)

    def item_json(it: ReviewItem, g: Game, namer: Namer, events: list[Event], video_url: Optional[str]) -> dict[str, Any]:
        e = it.event
        latest = _resolver(events)
        text = describe(e, namer, {}) or e.type.value
        if e.type == EventType.FLAG:
            text = str(e.payload.get("message") or text)
        return {
            "event_id": e.id, "seq": e.seq, "type": e.type.value, "team": e.team, "player_id": e.player_id,
            "text": when(e, g) + text, "confidence": e.confidence, "reason": it.reason,
            "impact": round(it.impact, 3), "video_ts": e.video_ts,
            "clip_start": it.clip_start, "clip_end": it.clip_end,
            "clip_seconds": round(it.clip_end - it.clip_start, 2),
            "video_url": video_url,
            "suggestions": [dict(index=i, label=s.label, action=s.action, changes=s.changes)
                            for i, s in enumerate(it.suggestions)],
            "event": e.model_dump(mode="json"),
        }

    def queue(g: Game, threshold: float) -> list[ReviewItem]:
        return correction_queue(store.events(g.id), st.rosters(g), threshold)

    @r.get("/games/{game_id}/review")
    def review(game_id: str, threshold: float = 0.8, p: Principal = Depends(require(*REVIEW_ROLES))):
        g = st.game(game_id)
        side = own_side(p, g)
        events = store.events(game_id)
        namer = Namer(g, st.team_names(g), st.players_map(g), p)
        video = st.media.get(game_id, {}).get("video_url")
        items = [it for it in queue(g, threshold) if side is None or it.event.team in (None, side)]
        return {"game_id": game_id, "threshold": threshold, "count": len(items), "video_url": video,
                "items": [item_json(it, g, namer, events, video) for it in items]}

    @r.post("/games/{game_id}/review/{event_id}")
    def review_action(game_id: str, event_id: str, body: ReviewIn,
                      p: Principal = Depends(require(*REVIEW_ROLES))):
        g = st.game(game_id)
        check_editable(g, p)
        side = own_side(p, g)
        ev = store.get_event(event_id)
        if ev is None or ev.game_id != game_id:
            raise HTTPException(404, "Event nicht gefunden")
        if event_id not in {e.id for e in effective_events(store.events(game_id))}:
            raise HTTPException(409, "Event wurde bereits korrigiert oder entfernt")
        if side is not None and ev.team not in (None, side):
            raise HTTPException(403, "Event gehört zur anderen Mannschaft")
        if body.action == "suggestion":
            items = {it.event.id: it for it in queue(g, 1.01)}
            it = items.get(event_id)
            if it is None or body.suggestion_index is None or not 0 <= body.suggestion_index < len(it.suggestions):
                raise HTTPException(400, "Ungültiger Vorschlagsindex")
            choice: Any = it.suggestions[body.suggestion_index]
        else:
            choice = body.action
        new = store.append(apply_review(ev, choice))
        return {"event": new.model_dump(mode="json"), "supersedes": event_id,
                "remaining": len(queue(g, 0.8))}

    @r.post("/games/{game_id}/fouls", status_code=201)
    def foul(game_id: str, body: FoulIn, p: Principal = Depends(require(*REVIEW_ROLES))):
        """Ein-Klick-Foul: Mannschaft, Viertel und Uhr werden aus Spieler und Tafel abgeleitet."""
        g = st.game(game_id)
        check_editable(g, p)
        pl = store.get_player(body.player_id)
        if pl is None:
            raise HTTPException(404, f"Spieler '{body.player_id}' nicht gefunden")
        try:
            side = g.side_of(pl.team_id)
        except KeyError:
            raise HTTPException(400, "Spieler gehört keinem der beiden Teams dieses Spiels an")
        mine = own_side(p, g)
        if mine is not None and mine != side:
            raise HTTPException(403, "Trainer dürfen nur Fouls der eigenen Mannschaft erfassen")
        evs = effective_events(store.events(game_id))
        period, clock, ts = body.period, body.clock, body.video_ts
        if period is None or clock is None:
            lp, lc = 0, None
            for e in reversed(evs):
                if e.type == EventType.SCOREBOARD and e.payload.get("period"):
                    lp, lc = int(e.payload["period"]), e.payload.get("clock")
                    break
                if e.period > 0:
                    lp, lc = e.period, e.clock
                    break
            period = lp if period is None else period
            clock = lc if clock is None else clock
        if ts is None:
            ts = evs[-1].video_ts if evs else 0.0
        ev = ConsoleFoulInput(game_id, store).foul(body.player_id, side, body.kind, body.fouled_id,  # type: ignore[arg-type]
                                                    video_ts=ts, period=period, clock=clock)
        box = st.box(g)
        return {"event": ev.model_dump(mode="json"), "player_fouls": box.players[body.player_id].pf
                if body.player_id in box.players else 1}

    @r.post("/games/{game_id}/scoresheet", status_code=201)
    def scoresheet(game_id: str, body: ScoresheetIn, p: Principal = Depends(require(*REVIEW_ROLES))):
        g = st.game(game_id)
        check_editable(g, p)
        payload: dict[str, Any] = {"players": body.players}
        if body.home is not None:
            payload["home"] = body.home
        if body.away is not None:
            payload["away"] = body.away
        events = store.events(game_id)
        for e in effective_events(events):  # frühere, unerledigte Abweichungen ersetzen
            if e.type == EventType.FLAG and e.payload.get("kind") == "scoresheet_mismatch" \
                    and e.status == EventStatus.PROVISIONAL:
                store.append(void(e, EventSource.SYSTEM, "durch neuen Spielbericht ersetzt"))
        ev = store.append(Event(
            game_id=game_id, type=EventType.SCORESHEET, source=EventSource.SCORESHEET,
            status=EventStatus.CONFIRMED, video_ts=events[-1].video_ts if events else 0.0, payload=payload))
        flags = [store.append(f) for f in reconcile_scoresheet(store.events(game_id), ev)]
        return {"event": ev.model_dump(mode="json"), "flags": [f.model_dump(mode="json") for f in flags],
                "mismatches": len(flags)}

    @r.post("/games/{game_id}/finalize")
    def do_finalize(game_id: str, body: FinalizeIn, p: Principal = Depends(require(*REVIEW_ROLES))):
        g = st.game(game_id)
        if g.status == GameStatus.FINAL:
            raise HTTPException(409, "Spiel ist bereits freigegeben")
        if body.force and p.role not in ("admin", "kampfgericht"):
            raise HTTPException(403, "Erzwungene Freigabe nur durch Kampfgericht oder Admin")
        try:
            ev = finalize(store, game_id, body.by or p.label, force=body.force)
        except FinalizationError as e:
            raise HTTPException(409, f"Freigabe verweigert: {e}")
        return {"event": ev.model_dump(mode="json"), "game": st.game(game_id).model_dump(mode="json")}

    return r
