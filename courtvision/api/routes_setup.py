"""Stammdaten und Setup (Hallenwart/Admin): Venues, Teams, Spieler, Spiele, Kalibrierung, Start/Ende."""
from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from ..domain.events import EventType
from ..domain.models import Game, GameStatus, Team, Venue
from ..geometry.court import LANDMARKS, Calibration, point_in_polygon
from .auth import Principal, current_principal, require
from .csvio import decode, parse_roster
from .geometry_data import court_geometry
from .schemas import CalibrationIn, EndIn, GameIn, MediaIn, PlayerIn, StartIn, TeamIn, VenueIn
from .state import AppState
from .visibility import player_view

SETUP = ("hallenwart",)
#: Mehr als diese mittlere Abweichung (Pixel) gilt die Kalibrierung als unsicher.
CALIB_WARN_PX = 8.0


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def build(st: AppState) -> APIRouter:
    r = APIRouter()
    store = st.store

    # ------------------------------------------------------------- Venues
    @r.post("/venues", status_code=201)
    def create_venue(body: VenueIn, _: Principal = Depends(require(*SETUP))):
        vid = body.id or _id("venue")
        if store.get_venue(vid):
            raise HTTPException(409, f"Halle '{vid}' existiert bereits")
        v = Venue(id=vid, name=body.name, city=body.city, scoreboard_source=body.scoreboard_source)
        store.put_venue(v)
        return v.model_dump()

    @r.get("/venues")
    def list_venues():
        return [v.model_dump() for v in store.list_venues()]

    # -------------------------------------------------------------- Teams
    @r.post("/teams", status_code=201)
    def create_team(body: TeamIn, _: Principal = Depends(require(*SETUP))):
        tid = body.id or _id("team")
        if store.get_team(tid):
            raise HTTPException(409, f"Team '{tid}' existiert bereits")
        t = Team(id=tid, **body.model_dump(exclude={"id"}))
        store.put_team(t)
        return t.model_dump()

    @r.get("/teams")
    def list_teams():
        return [t.model_dump() for t in store.list_teams()]

    @r.get("/teams/{team_id}")
    def get_team(team_id: str):
        return st.team(team_id).model_dump()

    @r.get("/teams/{team_id}/players")
    def team_players(team_id: str, p: Principal = Depends(current_principal)):
        st.team(team_id)
        return [player_view(p, pl) for pl in store.roster(team_id)]

    @r.post("/teams/{team_id}/roster.csv")
    async def import_roster(team_id: str, request: Request, _: Principal = Depends(require(*SETUP))):
        """CSV-Kaderimport. Body: rohes CSV (Content-Type text/csv). Spalten: number;name;[birth_year];[consent]."""
        st.team(team_id)
        raw = await request.body()
        if not raw.strip():
            raise HTTPException(400, "Leerer Request-Body: bitte CSV im Body senden")
        rows, errors = parse_roster(decode(raw))
        if not rows:
            raise HTTPException(422, {"message": "Keine gültigen Zeilen im CSV", "errors": errors})
        created = updated = 0
        for row in rows:
            _, new = st.upsert_player(team_id, row.number, row.name, row.birth_year, row.consent)
            created += int(new)
            updated += int(not new)
        roster = store.roster(team_id)
        return {
            "team_id": team_id, "imported": len(rows), "created": created, "updated": updated,
            "minors": sum(1 for p in roster if p.is_minor), "errors": errors,
            "players": [{"id": p.id, "number": p.number, "name": p.name, "is_minor": p.is_minor,
                         "consent": p.consent} for p in roster],
        }

    # ------------------------------------------------------------ Spieler
    @r.post("/players", status_code=201)
    def create_player(body: PlayerIn, _: Principal = Depends(require(*SETUP))):
        st.team(body.team_id)
        if body.id and store.get_player(body.id):
            raise HTTPException(409, f"Spieler '{body.id}' existiert bereits")
        p, _new = st.upsert_player(body.team_id, body.number, body.name, body.birth_year,
                                   body.consent or "", player_id=body.id)
        return p.model_dump()

    # -------------------------------------------------------------- Spiele
    @r.post("/games", status_code=201)
    def create_game(body: GameIn, _: Principal = Depends(require(*SETUP))):
        home, away = st.team(body.home_team_id), st.team(body.away_team_id)
        if home.id == away.id:
            raise HTTPException(400, "Heim- und Gastteam müssen verschieden sein")
        if body.venue_id and not store.get_venue(body.venue_id):
            raise HTTPException(404, f"Halle '{body.venue_id}' nicht gefunden")
        gid = body.id or _id("game")
        if store.get_game(gid):
            raise HTTPException(409, f"Spiel '{gid}' existiert bereits")
        data = body.model_dump(exclude={"id", "league", "age_class"})
        g = Game(id=gid, league=body.league if body.league is not None else home.league,
                 age_class=body.age_class if body.age_class is not None else home.age_class, **data)
        store.put_game(g)
        return g.model_dump(mode="json")

    @r.get("/games")
    def list_games(status: Optional[str] = None, league: Optional[str] = None):
        gs = [g for g in store.list_games()
              if (not status or g.status.value == status) and (not league or g.league == league)]
        gs.sort(key=lambda g: (g.start_time, g.id), reverse=True)
        names = {t.id: t.name for t in store.list_teams()}
        out = []
        for g in gs:
            d = g.model_dump(mode="json")
            d["home_name"], d["away_name"] = names.get(g.home_team_id, ""), names.get(g.away_team_id, "")
            out.append(d)
        return out

    @r.get("/games/{game_id}")
    def get_game(game_id: str):
        g = st.game(game_id)
        d = g.model_dump(mode="json")
        names = st.team_names(g)
        d["home_name"], d["away_name"] = names["home"], names["away"]
        return d

    @r.put("/games/{game_id}/media")
    def set_media(game_id: str, body: MediaIn, _: Principal = Depends(require(*SETUP))):
        st.game(game_id)
        return st.update_media(game_id, body.model_dump(exclude_unset=True))

    # ----------------------------------------------------------- Geometrie
    @r.get("/court/geometry")
    def geometry():
        return court_geometry()

    @r.post("/games/{game_id}/calibration")
    def calibrate(game_id: str, body: CalibrationIn, _: Principal = Depends(require(*SETUP))):
        st.game(game_id)
        unknown = [n for n in body.clicks if n not in LANDMARKS]
        if unknown:
            raise HTTPException(400, f"Unbekannte Landmarken: {', '.join(unknown)}")
        if len(body.clicks) < 4:
            raise HTTPException(400, "Mindestens 4 Landmarken nötig")
        zones = None
        if body.zones is not None:
            zones = {}
            for name, poly in body.zones.items():
                if len(poly) < 3:
                    raise HTTPException(400, f"Zone '{name}' braucht mindestens 3 Punkte")
                zones[name] = [(float(x), float(y)) for x, y in poly]
        try:
            cal = Calibration.from_landmarks({k: (float(v[0]), float(v[1])) for k, v in body.clicks.items()}, zones)
            err = cal.reprojection_error()
        except (ValueError, np.linalg.LinAlgError) as e:
            raise HTTPException(400, f"Kalibrierung nicht berechenbar: {e}")
        if not np.isfinite(err) or not np.all(np.isfinite(cal.H)):
            raise HTTPException(400, "Kalibrierung degeneriert (Punkte kollinear?)")
        info: dict[str, Any] = {
            "game_id": game_id, "n_points": len(body.clicks), "reprojection_error_px": err,
            "ok": err <= CALIB_WARN_PX,
            "message": "Kalibrierung gut" if err <= CALIB_WARN_PX else
            f"Mittlerer Fehler {err:.1f} px: bitte Punkte prüfen",
            "homography": cal.H.tolist(), "homography_inv": cal.H_inv.tolist(),
            "zones": {k: [list(pt) for pt in v] for k, v in (zones or {}).items()},
            "clicks": {k: list(v) for k, v in body.clicks.items()},
            "image_size": [body.image_width, body.image_height],
        }
        st.set_calibration(game_id, info)
        return info

    @r.get("/games/{game_id}/calibration")
    def get_calibration(game_id: str, _: Principal = Depends(require("hallenwart", "kampfgericht", "trainer", "system"))):
        st.game(game_id)
        info = st.calibration(game_id)
        if info is None:
            raise HTTPException(404, "Keine Kalibrierung gespeichert")
        return info

    @r.post("/games/{game_id}/calibration/zone-check")
    def zone_check(game_id: str, x: float, y: float, _: Principal = Depends(require("hallenwart", "system"))):
        """Welcher Zone liegt ein Feldpunkt (Meter)? Hilfsfunktion fuer Setup-Tests."""
        st.game(game_id)
        info = st.calibration(game_id)
        if not info:
            raise HTTPException(404, "Keine Kalibrierung gespeichert")
        zone = next((n for n, poly in info["zones"].items() if point_in_polygon(x, y, [tuple(p) for p in poly])), None)
        return {"zone": zone}

    # ---------------------------------------------------------- Start/Ende
    # Status-Lebenszyklus eines Spiels:
    #   scheduled --POST /start--> live --POST /end--> processing --POST /processed--> review --finalize--> final
    # * start: legt die GameSession an (Aufstellung, ggf. gespeicherte Wechselzonen) und schreibt
    #   GAME_START/STARTERS/PERIOD_START.
    # * end: session.finish() -- offene Ereignisse leeren, PERIOD_END/GAME_END, Status PROCESSING
    #   (Cloud-Nachverarbeitung laeuft).
    # * processed (system/admin): Nachverarbeitung fertig -> REVIEW. Die Korrekturliste
    #   (/review, /fouls, /scoresheet) ist in processing UND review bedienbar; /finalize ebenfalls.
    @r.post("/games/{game_id}/start")
    def start_game(game_id: str, body: StartIn, _: Principal = Depends(require(*SETUP))):
        g = st.game(game_id)
        if g.status != GameStatus.SCHEDULED:
            raise HTTPException(409, f"Spiel ist bereits '{g.status.value}'")
        rosters = st.rosters(g)
        for side in ("home", "away"):
            ids = body.starters.get(side, [])  # type: ignore[call-overload]
            if len(set(ids)) != 5:
                raise HTTPException(400, f"Genau 5 verschiedene Starter für '{side}' nötig")
            valid = {p.id for p in rosters[side]}
            bad = [i for i in ids if i not in valid]
            if bad:
                raise HTTPException(400, f"Spieler nicht im Kader von '{side}': {', '.join(bad)}")
        out = st.start_game(g, {s: list(body.starters[s]) for s in ("home", "away")}, body.video_ts)  # type: ignore[index]
        return {"game": st.game(game_id).model_dump(mode="json"), "events": [e.model_dump(mode="json") for e in out]}

    @r.post("/games/{game_id}/end")
    def end_game(game_id: str, body: Optional[EndIn] = None, _: Principal = Depends(require(*SETUP))):
        g = st.game(game_id)
        if g.status != GameStatus.LIVE:
            raise HTTPException(409, f"Nur laufende Spiele können beendet werden (Status '{g.status.value}')")
        slot = st.slot(game_id)
        with slot.lock:
            ts = body.video_ts if body and body.video_ts is not None else slot.session.last_ts
            out = slot.session.finish(ts)
        g = st.game(game_id)
        end_ev = next((e for e in out if e.type == EventType.GAME_END), None)
        return {"game": g.model_dump(mode="json"), "events": [e.model_dump(mode="json") for e in out],
                "event": end_ev.model_dump(mode="json") if end_ev else None}

    @r.post("/games/{game_id}/processed")
    def processed(game_id: str, _: Principal = Depends(require("system"))):
        """Cloud-Nachverarbeitung abgeschlossen: processing -> review."""
        g = st.game(game_id)
        if g.status != GameStatus.PROCESSING:
            raise HTTPException(409, f"Nur Spiele in 'processing' können übergeben werden (Status '{g.status.value}')")
        g = store.set_game_status(game_id, GameStatus.REVIEW)
        return {"game": g.model_dump(mode="json")}

    return r
