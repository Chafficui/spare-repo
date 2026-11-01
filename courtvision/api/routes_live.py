"""Events, WebSocket, Anzeigetafel-Ingest, Live-Stand, Box-Score, Shot Chart, Highlights."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from ..domain.events import Event, EventSource, EventStatus, EventType, effective_events
from ..domain.models import Game, GameStatus, Player
from ..highlights import (build_scenes, filter_publishable, live_clip, player_reel, recap, score_events)
from ..highlights.clips import ClipPlan, player_allowed
from ..highlights.overlay import name_allowed
from ..privacy import livestream_allowed
from ..scoreboard import ScoreboardReading
from ..stats import game_score, grade_boxscore
from .auth import Principal, current_principal, principal_from, require
from .schemas import EventIn, ScoreboardIn
from .state import AppState, Subscriber
from .text import Namer, fmt_clock, playbyplay
from .visibility import can_see_player, can_see_rating, event_visible_type, full_access, sanitize_event

COUNT_FIELDS = ("pts", "fgm", "fga", "fg2m", "fg2a", "fg3m", "fg3a", "ftm", "fta", "orb", "drb", "reb",
                "ast", "stl", "blk", "blka", "tov", "pf", "fd")
FINAL_LOCK_MSG = "Spiel ist freigegeben (final); Änderungen nur durch den Admin"


def highlight_channel(p: Principal, game: Game, requested: Optional[str] = None) -> str:
    """Ausspielkanal fuer den Einwilligungsfilter. Oeffentliche Rollen: nie weniger streng als 'public'."""
    if requested in ("public", "livestream"):
        return requested
    if p.operational or (p.role == "trainer" and p.team_id in (game.home_team_id, game.away_team_id)):
        return "team"
    return "public"


def build(st: AppState) -> APIRouter:
    r = APIRouter()
    store = st.store

    def latest_sb(events: list[Event]) -> Optional[Event]:
        sb = [e for e in effective_events(events) if e.type == EventType.SCOREBOARD]
        return sb[-1] if sb else None

    def serial_events(evs: list[Event], p: Principal, game: Game) -> list[dict[str, Any]]:
        pm = st.players_map(game)
        return [d for d in (sanitize_event(e, p, pm) for e in evs) if d is not None]

    def _text_for(ev: Event, game: Game, namer: Namer, all_events: Optional[list[Event]]) -> Optional[str]:
        from .text import SKIP, describe, when, _resolver
        if ev.type in SKIP:
            return None
        assists: dict[str, Event] = {}
        if all_events:
            latest = _resolver(all_events)
            assists = {latest(e.payload["shot_id"]): e for e in effective_events(all_events)
                       if e.type == EventType.ASSIST and e.payload.get("shot_id")}
        body = describe(ev, namer, assists)
        return None if body is None else when(ev, game) + body

    # ------------------------------------------------------------ Events
    @r.get("/games/{game_id}/events")
    def get_events(game_id: str, after_seq: int = 0, limit: int = Query(5000, ge=1, le=20000),
                   p: Principal = Depends(current_principal)):
        g = st.game(game_id)
        evs = store.events(game_id, after_seq)[:limit]
        return {"game_id": game_id, "events": serial_events(evs, p, g),
                "last_seq": evs[-1].seq if evs else after_seq}

    @r.post("/games/{game_id}/events", status_code=201)
    def post_event(game_id: str, body: EventIn, p: Principal = Depends(require("kampfgericht", "system"))):
        """Rohes Event anhängen. Menschliche Eingaben (kampfgericht/admin) sind standardmäßig
        ``console``/``confirmed``; die Pipeline (``system``) behält die Event-Standardwerte
        (``optical``/``provisional``)."""
        g = st.game(game_id)
        if g.status == GameStatus.FINAL and p.role != "admin":
            raise HTTPException(409, FINAL_LOCK_MSG)
        if body.supersedes:
            old = store.get_event(body.supersedes)
            if old is None or old.game_id != game_id:
                raise HTTPException(400, f"supersedes: Event '{body.supersedes}' gehört nicht zu diesem Spiel")
        data = body.model_dump(exclude_none=True, exclude={"payload"})
        data["payload"] = body.payload
        human = p.role != "system"
        data.setdefault("source", EventSource.CONSOLE if human else EventSource.OPTICAL)
        data.setdefault("status", EventStatus.CONFIRMED if human else EventStatus.PROVISIONAL)
        ev = store.append(Event(game_id=game_id, **data))
        derived: list[Event] = []
        if ev.source == EventSource.OPTICAL:  # optische Events laufen durch die Tafel-Fusion
            pipe = st.pipe(game_id)
            with pipe.lock:
                derived = [store.append(e) for e in pipe.fusion.on_optical_event(ev)]
        return {"event": ev.model_dump(mode="json"), "derived": [e.model_dump(mode="json") for e in derived]}

    @r.websocket("/ws/games/{game_id}")
    async def ws_game(ws: WebSocket, game_id: str, after_seq: int = 0, role: Optional[str] = None,
                      team_id: Optional[str] = None, player_id: Optional[str] = None):
        """Live-Events. Rolle per Query (Browser koennen keine Header setzen) oder ``X-Role``-Header
        -- gleicher Platzhalter wie in :mod:`courtvision.api.auth`."""
        h = ws.headers
        try:
            p = principal_from(role or h.get("x-role"), team_id or h.get("x-team-id"),
                               player_id or h.get("x-player-id"))
        except HTTPException:
            await ws.close(code=4401)
            return
        game = store.get_game(game_id)
        if game is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        sub = Subscriber(asyncio.get_running_loop())
        st.hub.add(game_id, sub)  # erst abonnieren, dann History lesen: nichts geht verloren
        try:
            pm = st.players_map(game)
            namer = Namer(game, st.team_names(game), pm, p)
            seen: set[int] = set()
            last = after_seq

            all_for_text = store.events(game_id)
            history = [e for e in all_for_text if e.seq > after_seq]
            for ev in history:
                seen.add(ev.seq)
                last = max(last, ev.seq)
                d = sanitize_event(ev, p, pm)
                if d is not None:
                    await ws.send_json({"kind": "event", "replay": True, "event": d,
                                        "text": _text_for(ev, game, namer, all_for_text)})
            await ws.send_json({"kind": "synced", "last_seq": last})

            async def pump() -> None:
                while True:
                    ev = await sub.queue.get()
                    if ev.seq in seen or ev.seq <= after_seq:
                        continue
                    seen.add(ev.seq)
                    pm_now = st.players_map(game)  # Einwilligungen koennen sich waehrend des Spiels aendern
                    d = sanitize_event(ev, p, pm_now)
                    if d is not None:
                        nm = Namer(game, st.team_names(game), pm_now, p)
                        await ws.send_json({"kind": "event", "replay": False, "event": d,
                                            "text": _text_for(ev, game, nm, store.events(game_id))})

            async def watch() -> None:
                while True:
                    await ws.receive_text()

            tasks = [asyncio.create_task(pump()), asyncio.create_task(watch())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except WebSocketDisconnect:
            pass
        finally:
            st.hub.remove(game_id, sub)

    # ------------------------------------------------------ Anzeigetafel
    @r.post("/games/{game_id}/scoreboard")
    def post_scoreboard(game_id: str, body: ScoreboardIn,
                        _: Principal = Depends(require("kampfgericht", "hallenwart", "system"))):
        """Anzeigetafel-Messwert: Plausibilitaetsfilter (mit Entprellung) -> Fusion -> Events."""
        g = st.game(game_id)
        if g.status == GameStatus.FINAL:
            raise HTTPException(409, FINAL_LOCK_MSG)
        reading = ScoreboardReading(**body.model_dump())
        pipe = st.pipe(game_id)
        with pipe.lock:
            out = pipe.filter.update(reading)
            events: list[Event] = []
            # Erst fusionieren, wenn beide Spielstaende entprellt uebernommen sind (ein SCOREBOARD-Event
            # ohne Punktestand wuerde die Box-Score-Projektion stoeren).
            if out.reading is not None and "home" in pipe.filter.acc and "away" in pipe.filter.acc:
                events = [store.append(e) for e in pipe.fusion.on_reading(out.reading)]
        return {
            "accepted": out.reading is not None, "changed": sorted(out.changed),
            "anomalies": [vars(a) for a in out.anomalies],
            "events": [e.model_dump(mode="json") for e in events],
        }

    # ---------------------------------------------------------------- Live
    @r.get("/games/{game_id}/live")
    def live(game_id: str, p: Principal = Depends(current_principal)):
        g = st.game(game_id)
        events = store.events(game_id)
        box = st.box(g, events)
        sb = latest_sb(events)
        names = st.team_names(g)
        pm = st.players_map(g)
        namer = Namer(g, names, pm, p)
        pbp = playbyplay(events, g, namer, p, limit=20)
        period = clock = None
        running = False
        fouls: dict[str, Any] = {"home": None, "away": None}
        if sb is not None:
            period, clock = sb.payload.get("period") or sb.period or None, sb.payload.get("clock")
            running = bool(sb.payload.get("running"))
            fouls = {"home": sb.payload.get("fouls_home"), "away": sb.payload.get("fouls_away")}
        else:
            for e in reversed(effective_events(events)):
                if e.period > 0:
                    period, clock = e.period, e.clock
                    break
        for side in ("home", "away"):  # Fallback: Fouls aus dem Box-Score (aktuelles Viertel)
            if fouls[side] is None:
                tf = box.team(side).team_fouls_by_period  # type: ignore[arg-type]
                fouls[side] = tf.get(period or 0, 0) if tf or period else 0
        allowed = livestream_allowed(g, pm.values())
        media = st.media.get(game_id, {})
        return {
            "game_id": game_id, "status": g.status.value, "league": g.league,
            "home": {"team_id": g.home_team_id, "name": names["home"], "score": box.score["home"]},
            "away": {"team_id": g.away_team_id, "name": names["away"], "score": box.score["away"]},
            "score": box.score, "score_mismatch": box.score_mismatch,
            "period": period, "clock": clock, "clock_text": fmt_clock(clock) if clock is not None else None,
            "running": running, "team_fouls": fouls,
            "box_status": box.status,
            "livestream": {"allowed": allowed, "url": media.get("stream_url") if allowed else None},
            "playbyplay": list(reversed(pbp)),
            "last_seq": events[-1].seq if events else 0,
        }

    @r.get("/games/{game_id}/playbyplay")
    def get_pbp(game_id: str, limit: int = Query(200, ge=1, le=5000), p: Principal = Depends(current_principal)):
        g = st.game(game_id)
        events = store.events(game_id)
        namer = Namer(g, st.team_names(g), st.players_map(g), p)
        return {"game_id": game_id, "items": list(reversed(playbyplay(events, g, namer, p, limit=limit)))}

    # ------------------------------------------------------------ Box-Score
    @r.get("/games/{game_id}/boxscore")
    def boxscore(game_id: str, p: Principal = Depends(current_principal)):
        g = st.game(game_id)
        events = store.events(game_id)
        box = st.box(g, events)
        pm = st.players_map(g)
        dist_games = st.league_games(g.league) if g.league else [g]
        grades = grade_boxscore(box, _dist(st, g, dist_games))
        names = st.team_names(g)
        lines, hidden = [], 0
        for pid, ln in box.players.items():
            pl = pm.get(pid)
            active = ln.seconds_played > 0 or any(getattr(ln, f) for f in COUNT_FIELDS)
            if not can_see_player(p, pl):
                hidden += int(active)
                continue
            d = ln.model_dump(exclude={"shots"})
            d.update(minutes=round(ln.minutes, 2), efg_pct=ln.efg_pct, ts_pct=ln.ts_pct,
                     name=pl.name if pl else ln.name, team_id=pl.team_id if pl else g.team_id(ln.team))
            gr = grades.get(pid)
            if gr is not None and can_see_rating(p, pl):
                d["rating"] = {"grade": gr.value, "provisional": gr.provisional, "pir": gr.pir, "eff": gr.eff,
                               "game_score": round(gr.raw_gmsc, 1), "percentile": round(gr.percentile, 3)}
            else:
                d["rating"] = None
            lines.append(d)
        lines.sort(key=lambda d: (d["team"], d["number"]))

        def team_json(side: str) -> dict[str, Any]:
            t = box.team(side)  # type: ignore[arg-type]
            tot = t.totals
            return {"side": side, "team_id": g.team_id(side), "name": names[side],  # type: ignore[arg-type]
                    "points": t.points, "totals": {f: getattr(tot, f) for f in COUNT_FIELDS} | {
                        "efg_pct": tot.efg_pct, "ts_pct": tot.ts_pct},
                    "team_rebounds": t.team_rebounds, "team_turnovers": t.team_turnovers,
                    "team_fouls_by_period": {str(k): v for k, v in t.team_fouls_by_period.items()}}

        lineups = []
        for lu in box.lineups:
            if not full_access(p, g.team_id(lu.side)):  # type: ignore[arg-type]
                continue
            lineups.append({
                "side": lu.side, "seconds": lu.seconds, "pts_for": lu.pts_for, "pts_against": lu.pts_against,
                "net": lu.net,
                "players": [{"player_id": x, "number": pm[x].number if x in pm else None,
                             "name": pm[x].name if x in pm else x} for x in lu.players]})
        lineups.sort(key=lambda d: -d["seconds"])
        view = "full" if p.operational else ("team" if p.role == "trainer" else "public")
        return {"game_id": game_id, "status": box.status, "game_status": g.status.value, "score": box.score,
                "score_mismatch": box.score_mismatch, "home": team_json("home"), "away": team_json("away"),
                "players": lines, "hidden_players": hidden, "lineups": lineups, "view": view,
                "note": ("Spieler ohne Freigabe werden nur als Team-Wert ausgewiesen." if hidden else "")}

    # ------------------------------------------------------------ Shot Chart
    @r.get("/games/{game_id}/shotchart")
    def shotchart(game_id: str, player_id: Optional[str] = None, team: Optional[str] = None,
                  p: Principal = Depends(current_principal)):
        g = st.game(game_id)
        box = st.box(g)
        pm = st.players_map(g)
        if player_id:
            if player_id not in box.players and player_id not in pm:
                raise HTTPException(404, f"Spieler '{player_id}' nicht gefunden")
            if not can_see_player(p, pm.get(player_id)):
                raise HTTPException(403, "Keine Freigabe für Einzelstatistiken dieses Spielers")
        shots = []
        for pid, ln in box.players.items():
            if player_id and pid != player_id:
                continue
            if team in ("home", "away") and ln.team != team:
                continue
            show = can_see_player(p, pm.get(pid))
            for s in ln.shots:
                if s.get("x") is None or s.get("y") is None:
                    continue
                shots.append({"x": s["x"], "y": s["y"], "value": s["value"], "made": s["made"],
                              "team": ln.team, "player_id": pid if show else None,
                              "number": ln.number if show else None, "event_id": s.get("event_id")})
        n, m = len(shots), sum(1 for s in shots if s["made"])
        return {"game_id": game_id, "player_id": player_id, "shots": shots,
                "summary": {"fga": n, "fgm": m, "fg_pct": (m / n) if n else None,
                            "fg3a": sum(1 for s in shots if s["value"] == 3),
                            "fg3m": sum(1 for s in shots if s["value"] == 3 and s["made"])}}

    # ------------------------------------------------------------ Highlights
    def _cand(s, pm: dict[str, Player], namer: Namer, game: Game, events: list[Event]) -> dict[str, Any]:
        e = s.event
        pl = pm[e.player_id]  # type: ignore[index]
        return {"event_id": e.id, "type": e.type.value, "score": round(s.score, 2), "player_id": pl.id,
                "number": pl.number, "name": pl.name if name_allowed(pl) else None, "team": e.team,
                "period": e.period, "clock": e.clock, "video_ts": e.video_ts,
                "clip_start": e.clip_start if e.clip_start is not None else max(0.0, e.video_ts - 6.0),
                "clip_end": e.clip_end if e.clip_end is not None else e.video_ts + 3.0,
                "text": _text_for(e, game, namer, events), "close_game": s.context.close_game,
                "clutch": s.context.clutch_time}

    @r.get("/games/{game_id}/highlights")
    def highlights(game_id: str, channel: Optional[str] = None, p: Principal = Depends(current_principal)):
        g = st.game(game_id)
        events = store.events(game_id)
        pm = st.players_map(g)
        ch = highlight_channel(p, g, channel)
        namer = Namer(g, st.team_names(g), pm, p)
        cands = [_cand(s, pm, namer, g, events) for s in score_events(events, g)
                 if s.event.player_id and player_allowed(pm.get(s.event.player_id), ch)]
        scenes = [s for s in build_scenes(events, g) if s.actor_id and player_allowed(pm.get(s.actor_id), ch)]
        plans = [x for x in (recap(scenes, pm), live_clip(scenes, pm)) if x is not None]
        plans = filter_publishable(plans, pm, ch)  # zweite Sicherung
        return {"game_id": game_id, "channel": ch, "candidates": cands[:30],
                "plans": [dict(pl.model_dump(mode="json"), duration=pl.duration) for pl in plans],
                "recap": next((dict(pl.model_dump(mode="json"), duration=pl.duration)
                               for pl in plans if pl.kind == "recap"), None),
                "ai_label": "KI-generiert"}

    return r


def _dist(st: AppState, g: Game, dist_games: list[Game]):
    """Ligaverteilung (GmSc/40) aus den uebrigen Spielen der Liga; <20 Werte -> Fallback."""
    from ..stats import LeagueDistribution
    from .state import _game_date
    dist = LeagueDistribution()
    for dg in dist_games:
        b = st.box(dg)
        d = _game_date(dg)
        for ln in b.players.values():
            if ln.seconds_played >= 300:
                dist.add(d, game_score(ln) * 2400.0 / ln.seconds_played)
    return dist
