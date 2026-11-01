"""Spielerprofil, Reels, Bestenliste, Einwilligungen (Datenschutz), DSFA, Verbands-Export, Abrechnung."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from ..business.pricing import PricingPlan, invoice
from ..domain.models import Consent, Player
from ..highlights import filter_publishable, player_reel
from ..highlights.clips import ClipPlan
from ..privacy import can_show_stats_public, public_ratings_allowed
from ..privacy.dpia_checklist import ItemStatus
from ..stats import leaderboard, season_profile
from .auth import Principal, current_principal, require
from .csvio import to_csv
from .routes_live import COUNT_FIELDS, highlight_channel
from .schemas import ConsentIn, DpiaUpdate
from .state import AppState, _game_date
from .visibility import can_see_player, can_see_rating, can_see_private_fields, full_access, is_self

LEVEL_NAMES = {"STATS_INTERNAL": Consent.STATS_INTERNAL, "STATS_PUBLIC": Consent.STATS_PUBLIC,
               "HIGHLIGHTS": Consent.HIGHLIGHTS, "LIVESTREAM": Consent.LIVESTREAM,
               "TRAINING_DATA": Consent.TRAINING_DATA}


def consent_names(value: int) -> list[str]:
    c = Consent(value)
    if Consent.OBJECTED in c:
        return ["OBJECTED"]
    return [n for n, f in LEVEL_NAMES.items() if f in c]


def build(st: AppState) -> APIRouter:
    r = APIRouter()
    store = st.store

    def profile_payload(pl: Player, p: Principal, with_games: bool = True) -> dict[str, Any]:
        team = st.team(pl.team_id)
        games = st.team_games(pl.team_id)
        league_games = st.league_games(team.league) if team.league else games
        data = st.collect(games, league_games).get(pl.id, [])
        lines = [x[1] for x in data]
        grades = [x[2].value for x in data]
        see_rating = can_see_rating(p, pl)
        prof = season_profile(lines, grades if see_rating else [], pl.id)
        out: dict[str, Any] = {
            "player": {"id": pl.id, "team_id": pl.team_id, "team_name": team.name, "number": pl.number,
                       "name": pl.name},
            "games": prof.games, "minutes": round(prof.minutes, 1), "radar": prof.radar, "bests": prof.bests,
            "rating_visible": see_rating,
            "average_grade": prof.average_grade if see_rating else None,
            "form": prof.form if see_rating else [],
            "provisional": any(x[2].provisional for x in data) if see_rating else None,
        }
        if not see_rating:
            out["bests"] = {k: v for k, v in prof.bests.items() if k != "grade"}
        if with_games:
            names = {t.id: t.name for t in store.list_teams()}
            rows = []
            for g, ln, gr in data:
                opp = g.away_team_id if pl.team_id == g.home_team_id else g.home_team_id
                row = {"game_id": g.id, "date": g.start_time[:10], "status": g.status.value,
                       "opponent": names.get(opp, opp), "minutes": round(ln.minutes, 1)}
                row.update({f: getattr(ln, f) for f in COUNT_FIELDS})
                row["grade"] = gr.value if see_rating else None
                row["provisional"] = gr.provisional if see_rating else None
                rows.append(row)
            out["per_game"] = rows
        return out

    # ----------------------------------------------------------------- Profil
    @r.get("/players/{player_id}")
    def get_player(player_id: str, p: Principal = Depends(current_principal)):
        from .visibility import player_view
        return player_view(p, st.player(player_id))

    @r.get("/players/{player_id}/profile")
    def profile(player_id: str, p: Principal = Depends(current_principal)):
        pl = st.player(player_id)
        if not can_see_player(p, pl):
            raise HTTPException(403, "Der Spieler hat der Veröffentlichung seiner Statistiken nicht zugestimmt")
        return profile_payload(pl, p)

    @r.get("/players/{player_id}/reel")
    def reel(player_id: str, channel: Optional[str] = None, p: Principal = Depends(current_principal)):
        pl = st.player(player_id)
        segs = []
        ch = "public"
        for g in st.team_games(pl.team_id):
            ch = highlight_channel(p, g, channel) if not is_self(p, pl.id) else (
                channel if channel in ("public", "livestream") else "team")
            pm = st.players_map(g)
            plan = player_reel(store.events(g.id), pl.id, g, players=pm)
            kept = filter_publishable([plan], pm, ch) if plan is not None else []
            if kept:
                segs += kept[0].segments
        out: dict[str, Any] = {"player_id": pl.id, "channel": ch, "plan": None, "ai_label": "KI-generiert"}
        if segs:
            plan = ClipPlan(kind="player_reel", segments=segs)
            out["plan"] = dict(plan.model_dump(mode="json"), duration=plan.duration)
        else:
            out["message"] = "Keine freigegebenen Szenen"
        return out

    @r.get("/leagues/{league}/leaderboard")
    def league_leaderboard(league: str, metric: str = "average_grade", limit: int = Query(50, ge=1, le=500),
                           min_games: int = Query(1, ge=1)):
        """Bestenliste (nur freigegebene Spiele). Minderjaehrige und Spieler ohne oeffentliche Freigabe
        werden nie gelistet."""
        games = st.league_games(league, final_only=True)
        data = st.collect(games)
        players = {}
        for g in games:
            players.update(st.players_map(g))
        excluded = {pid for pid in data if pid not in players or players[pid].is_minor
                    or not public_ratings_allowed(players[pid])}
        profs = [season_profile([x[1] for x in rows], [x[2].value for x in rows], pid) for pid, rows in data.items()]
        try:
            ranked = leaderboard(profs, metric, minors=excluded, limit=limit, min_games=min_games)
        except KeyError:
            raise HTTPException(400, f"Unbekannte Kennzahl '{metric}'")
        teams = {t.id: t.name for t in store.list_teams()}
        return {"league": league, "metric": metric, "games": len(games), "rows": [
            {"rank": i + 1, "player_id": pr.player_id, "number": players[pr.player_id].number,
             "name": players[pr.player_id].name, "team": teams.get(players[pr.player_id].team_id, ""),
             "games": pr.games, "minutes": round(pr.minutes, 1), "average_grade": pr.average_grade,
             "form": pr.form, "bests": pr.bests, "radar": pr.radar}
            for i, pr in enumerate(ranked)]}

    # ----------------------------------------------------------- Einwilligung
    def consent_guard(p: Principal, pl: Player) -> None:
        if not (p.role in ("admin", "hallenwart") or is_self(p, pl.id)):
            raise HTTPException(403, "Einwilligungen ändern dürfen nur der Spieler/die Eltern, der Verein oder Admin")

    def consent_json(player_id: str, with_audit: bool) -> dict[str, Any]:
        pl = st.ensure_registered(player_id)
        out = {"player_id": player_id, "is_minor": pl.is_minor, "effective": pl.consent,
               "levels": consent_names(pl.consent), "objected": bool(pl.consent & int(Consent.OBJECTED)),
               "stats_public": can_show_stats_public(pl), "public_ratings": public_ratings_allowed(pl),
               "needs_child_consent": st.cm.needs_child_consent(player_id) if pl.is_minor else False}
        if with_audit:
            out["audit"] = [rec.model_dump() for rec in st.cm.audit_trail(player_id)]
        return out

    @r.get("/players/{player_id}/consent")
    def get_consent(player_id: str, p: Principal = Depends(current_principal)):
        pl = st.player(player_id)
        if not (p.role in ("admin", "hallenwart") or is_self(p, pl.id) or full_access(p, pl.team_id)):
            raise HTTPException(403, "Keine Berechtigung für Einwilligungsdaten")
        return consent_json(player_id, True)

    @r.post("/players/{player_id}/consent")
    def change_consent(player_id: str, body: ConsentIn, p: Principal = Depends(current_principal)):
        pl = st.player(player_id)
        consent_guard(p, pl)
        actor = {"spieler": "player", "hallenwart": "club", "admin": "club"}[p.role]
        levels = Consent.NONE
        for name in body.level or []:
            if name not in LEVEL_NAMES:
                raise HTTPException(400, f"Unbekannte Stufe '{name}' (erlaubt: {', '.join(LEVEL_NAMES)})")
            levels |= LEVEL_NAMES[name]
        a = body.action
        if a in ("grant", "revoke", "parental_consent") and not levels:
            raise HTTPException(400, "Für diese Aktion ist 'level' nötig")
        with st.lock:
            st.ensure_registered(player_id)
            cm = st.cm
            if a == "grant":
                rec = cm.grant(player_id, levels, actor=actor, note=body.note)
            elif a == "revoke":
                rec = cm.revoke(player_id, levels, actor=actor, note=body.note)
            elif a == "object":
                rec = cm.object_(player_id, actor=actor, note=body.note)
            elif a == "withdraw_objection":
                rec = cm.withdraw_objection(player_id, actor=actor, note=body.note)
            elif a == "parental_consent":
                rec = cm.grant_parental(player_id, levels, note=body.note)
            elif a == "revoke_parental":
                rec = cm.revoke_parental(player_id, note=body.note)
            elif a == "child_consent":
                rec = cm.grant_child(player_id, note=body.note)
            else:
                rec = cm.revoke_child(player_id, note=body.note)
            st.persist_player(player_id)
        return {"record": rec.model_dump(), "consent": consent_json(player_id, False)}

    # ------------------------------------------------------------------ DSFA
    def dpia_json() -> dict[str, Any]:
        cl = st.dpia
        return {"items": [i.model_dump(mode="json") for i in cl.items.values()], "progress": cl.progress(),
                "complete": cl.is_complete(), "pilot_blockers": [i.id for i in cl.pilot_blockers()]}

    @r.get("/privacy/dpia")
    def dpia(_: Principal = Depends(require("hallenwart", "verband"))):
        return dpia_json()

    @r.post("/privacy/dpia/{item_id}")
    def dpia_update(item_id: str, body: DpiaUpdate, _: Principal = Depends(require())):
        if item_id not in st.dpia.items:
            raise HTTPException(404, "Unbekannter Punkt")
        st.set_dpia(item_id, ItemStatus(body.status), body.owner, body.note)
        return dpia_json()

    # ---------------------------------------------------------------- Verband
    @r.get("/leagues/{league}/export.csv")
    def export_csv(league: str, delimiter: str = ";", _: Principal = Depends(require("verband"))):
        """Box-Score-Zeilen aller freigegebenen Spiele der Liga. Einzelzeilen nur fuer Spieler mit
        oeffentlicher Freigabe; alle uebrigen sind nur in der Team-Summenzeile enthalten. Keine Ratings."""
        if delimiter not in (";", ",", "\t"):
            raise HTTPException(400, "delimiter muss ';', ',' oder Tab sein")
        header = ["game_id", "date", "home", "away", "team", "number", "name", "min", *COUNT_FIELDS, "plus_minus"]
        rows = []
        for g in st.league_games(league, final_only=True):
            box = st.box(g)
            pm = st.players_map(g)
            names = st.team_names(g)
            common = [g.id, g.start_time[:10], names["home"], names["away"]]
            for side in ("home", "away"):
                for ln in sorted((x for x in box.players.values() if x.team == side), key=lambda x: x.number):
                    pl = pm.get(ln.player_id)
                    if pl is None or not can_show_stats_public(pl):
                        continue
                    rows.append(common + [names[side], ln.number, pl.name, round(ln.minutes, 1)]
                                + [getattr(ln, f) for f in COUNT_FIELDS] + [ln.plus_minus])
                tot = box.team(side).totals  # type: ignore[arg-type]
                rows.append(common + [names[side], "", "TEAM", ""] + [getattr(tot, f) for f in COUNT_FIELDS] + [""])
        return Response(to_csv(header, rows, delimiter), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{league}-export.csv"'})

    # --------------------------------------------------------------- Business
    @r.get("/pricing/invoice")
    def pricing_invoice(games: int = Query(..., ge=0, le=100000),
                        _: Principal = Depends(require("hallenwart", "verband"))):
        inv = invoice(games, PricingPlan())
        return {"games": games, "lines": [{"label": l, "amount": a} for l, a in inv.lines],
                "net": inv.net, "vat": inv.vat, "total": inv.total, "currency": "EUR"}

    return r
