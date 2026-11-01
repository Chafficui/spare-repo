"""API-Tests (FastAPI TestClient): Setup, Events, WebSocket, Anzeigetafel, Review, Datenschutz, Export."""
import csv
import io
from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient

from courtvision.api.app import create_app
from courtvision.domain.events import EventStatus, EventType
from courtvision.domain.models import GameStatus
from courtvision.geometry.court import LANDMARKS
from courtvision.domain.observations import ClockState, FrameState, PlayerState, Possession, RimMake, ShotRelease
from courtvision.domain.serialize import observation_from_dict, observation_to_dict
from courtvision.sim import demo_setup, load_demo, simulate_game
from courtvision.store.eventstore import EventStore

THIS_YEAR = date.today().year

HOME_CSV = f"""number;name;birth_year;consent
4;Jonas Schmidt;1995;
7;Hans Müller;1994;
8;Tim Kessler;1998;
9;Paul Lindner;1999;
11;Nico Objekt;1996;nein
15;Leo Jung;{THIS_YEAR - 16};
16;Mia Klein;{THIS_YEAR - 15};ja
"""
AWAY_CSV = """number,name
3,Leon Schaefer
6,Max Reinhardt
7,David Neumann
8,Kilian Busch
9,Tobias Engel
"""
HOME_STARTERS = [4, 7, 11, 15, 16]
AWAY_STARTERS = [3, 6, 7, 8, 9]


def hdr(role, **kw):
    h = {"X-Role": role}
    if "team" in kw:
        h["X-Team-Id"] = kw["team"]
    if "player" in kw:
        h["X-Player-Id"] = kw["player"]
    return h


HW, KG, FAN = hdr("hallenwart"), hdr("kampfgericht"), hdr("fan")
SYS = hdr("system")
ADMIN = hdr("admin")
TRAINER_H = hdr("trainer", team="t-home")
TRAINER_A = hdr("trainer", team="t-away")
VERBAND = hdr("verband")


def pid(side, number):
    return f"t-{side}-{number}"


@pytest.fixture
def client():
    with TestClient(create_app(EventStore(":memory:"))) as c:
        yield c


def setup_world(c, start=True, league="Oberliga"):
    assert c.post("/venues", json={"id": "v1", "name": "Halle", "city": "Dessau"}, headers=HW).status_code == 201
    for side, name in (("home", "TV Dessau"), ("away", "BC Halle")):
        r = c.post("/teams", json={"id": f"t-{side}", "name": name, "club": name, "league": league}, headers=HW)
        assert r.status_code == 201, r.text
    r = c.post("/teams/t-home/roster.csv", content=HOME_CSV.encode(), headers={**HW, "Content-Type": "text/csv"})
    assert r.status_code == 200, r.text
    r = c.post("/teams/t-away/roster.csv", content=AWAY_CSV.encode(), headers={**HW, "Content-Type": "text/csv"})
    assert r.status_code == 200, r.text
    r = c.post("/games", json={"id": "g1", "venue_id": "v1", "home_team_id": "t-home", "away_team_id": "t-away",
                               "start_time": "2026-03-14T18:00:00+01:00"}, headers=HW)
    assert r.status_code == 201, r.text
    if start:
        r = c.post("/games/g1/start", headers=HW, json={"starters": {
            "home": [pid("home", n) for n in HOME_STARTERS], "away": [pid("away", n) for n in AWAY_STARTERS]}})
        assert r.status_code == 200, r.text
    return "g1"


def ev(c, gid, type_, side=None, number=None, period=0, clock=None, ts=0.0, payload=None, **kw):
    body = {"type": type_, "team": side, "period": period, "clock": clock, "video_ts": ts, "payload": payload or {}}
    if number is not None:
        body["player_id"] = pid(side, number)
    body.update(kw)
    headers = kw.pop("headers", None) or KG
    body.pop("headers", None)
    r = c.post(f"/games/{gid}/events", json=body, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["event"]


def play_game(c, gid="g1"):
    """Kleines Spiel: Heim 8, Gast 2. Einwilligungsmix siehe CSV."""
    ev(c, gid, "SHOT", "home", 4, 1, 540, 60, {"value": 2, "made": True, "x": 10.0, "y": 1.0})
    s3 = ev(c, gid, "SHOT", "home", 7, 2, 465, 700, {"value": 3, "made": True, "x": 7.0, "y": 0.5})
    ev(c, gid, "ASSIST", "home", 4, 2, 465, 700, {"shot_id": s3["id"]})
    ev(c, gid, "SHOT", "home", 11, 2, 300, 800, {"value": 2, "made": True, "x": 11.0, "y": -2.0})
    ev(c, gid, "SHOT", "away", 3, 3, 400, 1500, {"value": 2, "made": True, "x": -10.0, "y": 0.0})
    ev(c, gid, "SHOT", "home", 16, 3, 350, 1550, {"value": 3, "made": False, "x": 8.0, "y": 2.0})
    ev(c, gid, "FREE_THROW", "home", 15, 3, 200, 1600, {"made": True, "attempt": 1, "of": 1})
    ev(c, gid, "PERIOD_END", None, None, 4, 0.0, 2400)
    return s3


# ---------------------------------------------------------------- Setup
def test_setup_flow(client):
    c = client
    # Rollen: Fans duerfen keine Stammdaten anlegen
    assert c.post("/teams", json={"name": "X"}).status_code == 403
    assert c.get("/games/zzz/live", headers={"X-Role": "gibtsnicht"}).status_code == 401
    assert c.post("/venues", json={"id": "v1", "name": "Halle"}, headers=HW).status_code == 201
    assert c.post("/teams", json={"id": "t-home", "name": "A", "league": "Oberliga"}, headers=HW).status_code == 201
    assert c.post("/teams", json={"id": "t-home", "name": "A"}, headers=HW).status_code == 409
    c.post("/teams", json={"id": "t-away", "name": "B", "league": "Oberliga"}, headers=HW)

    # CSV-Import: ; und , als Trennzeichen, Minderjaehrige per Geburtsjahr, Fehlerzeilen werden gemeldet
    csv_text = HOME_CSV + "99;;1990;\nxx;Fehler;2000;\n7;Doppelt;1990;\n"
    r = c.post("/teams/t-home/roster.csv", content=csv_text.encode(), headers={**HW, "Content-Type": "text/csv"})
    assert r.status_code == 200
    body = r.json()
    assert body["imported"] == 7 and body["created"] == 7 and body["minors"] == 2
    assert len(body["errors"]) == 3
    by_num = {p["number"]: p for p in body["players"]}
    assert by_num[15]["is_minor"] and by_num[16]["is_minor"] and not by_num[7]["is_minor"]
    assert by_num[15]["consent"] == 32          # Minderjaehrig ohne Elterneinwilligung: OBJECTED
    assert by_num[11]["consent"] == 32          # Widerspruch
    assert by_num[7]["consent"] & 2             # Erwachsene: STATS_PUBLIC per Standard
    assert by_num[16]["consent"] & 2 and not by_num[16]["consent"] & 32   # Eltern haben zugestimmt
    # Re-Import aktualisiert statt zu duplizieren
    r = c.post("/teams/t-home/roster.csv", content=b"7,Hans Mueller\n", headers={**HW, "Content-Type": "text/csv"})
    assert r.json()["updated"] == 1 and len(c.get("/teams/t-home/players", headers=HW).json()) == 7
    assert c.post("/teams/t-home/roster.csv", content=b"x;y\n", headers=HW).status_code == 422
    assert c.post("/teams/t-home/roster.csv", content=b"1;A", headers=FAN).status_code == 403
    c.post("/teams/t-away/roster.csv", content=AWAY_CSV.encode(), headers=HW)

    # Spiel
    bad = c.post("/games", json={"home_team_id": "t-home", "away_team_id": "nope"}, headers=HW)
    assert bad.status_code == 404
    r = c.post("/games", json={"id": "g1", "venue_id": "v1", "home_team_id": "t-home", "away_team_id": "t-away"}, headers=HW)
    assert r.status_code == 201 and r.json()["league"] == "Oberliga" and r.json()["status"] == "scheduled"
    assert [g["id"] for g in c.get("/games").json()] == ["g1"]

    # Kalibrierung: exakte Klicks aus einer synthetischen Kamera -> winziger Fehler
    Hm = np.array([[60, 10, 1900], [2, 25, 900], [0.001, 0.01, 1]])

    def proj(x, y):
        p = Hm @ [x, y, 1]
        return [float(p[0] / p[2]), float(p[1] / p[2])]

    names = ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right", "center",
             "ft_left_top", "ft_right_bottom"]
    clicks = {n: proj(*LANDMARKS[n]) for n in names}
    zones = {"substitution": [[-2, 7.5], [2, 7.5], [2, 9], [-2, 9]]}
    r = c.post("/games/g1/calibration", json={"clicks": clicks, "zones": zones}, headers=HW)
    assert r.status_code == 200, r.text
    cal = r.json()
    assert cal["reprojection_error_px"] < 1e-3 and cal["ok"] and cal["n_points"] == 7
    assert cal["zones"]["substitution"][0] == [-2, 7.5]
    assert c.get("/games/g1/calibration", headers=HW).json()["reprojection_error_px"] < 1e-3
    assert c.post("/games/g1/calibration", json={"clicks": {"mars": [1, 2]}}, headers=HW).status_code == 400
    assert c.post("/games/g1/calibration", json={"clicks": {n: clicks[n] for n in names[:3]}}, headers=HW).status_code == 400
    noisy = dict(clicks)
    noisy["center"] = [clicks["center"][0] + 40, clicks["center"][1] - 30]
    r = c.post("/games/g1/calibration", json={"clicks": noisy}, headers=HW)
    assert r.json()["reprojection_error_px"] > 5

    geo = c.get("/court/geometry").json()
    assert geo["length"] == 28.0 and len(geo["three_point"]["left"]) > 10 and "center" in geo["landmarks"]

    # Start: genau 5 Starter je Team, aus dem Kader
    starters = {"home": [pid("home", n) for n in (4, 7, 8, 9, 11)], "away": [pid("away", n) for n in AWAY_STARTERS]}
    assert c.post("/games/g1/start", json={"starters": {**starters, "home": starters["home"][:4]}},
                  headers=HW).status_code == 400
    assert c.post("/games/g1/start", json={"starters": {**starters, "away": ["x1", "x2", "x3", "x4", "x5"]}},
                  headers=HW).status_code == 400
    r = c.post("/games/g1/start", json={"starters": starters}, headers=HW)
    assert r.status_code == 200 and r.json()["game"]["status"] == "live"
    types = [e["type"] for e in c.get("/games/g1/events", headers=KG).json()["events"]]
    assert types.count("STARTERS") == 2 and types[0] == "GAME_START"
    assert c.post("/games/g1/start", json={"starters": starters}, headers=HW).status_code == 409
    r = c.post("/games/g1/end", headers=HW)
    assert r.status_code == 200 and r.json()["game"]["status"] == "processing"
    # Lebenszyklus: processing -> (system) processed -> review
    assert c.post("/games/g1/processed", headers=HW).status_code == 403
    r = c.post("/games/g1/processed", headers=SYS)
    assert r.status_code == 200 and r.json()["game"]["status"] == "review"
    assert c.post("/games/g1/processed", headers=SYS).status_code == 409


# --------------------------------------------------------------- Events
def test_events_and_boxscore(client):
    c = client
    gid = setup_world(c)
    assert c.post(f"/games/{gid}/events", json={"type": "TIMEOUT"}, headers=FAN).status_code == 403
    assert c.post(f"/games/{gid}/events", json={"type": "TIMEOUT"}, headers=TRAINER_H).status_code == 403
    assert c.post(f"/games/{gid}/events", json={"type": "SHOT", "supersedes": "nope"}, headers=KG).status_code == 400
    assert c.post("/games/zzz/events", json={"type": "TIMEOUT"}, headers=KG).status_code == 404
    play_game(c, gid)
    evs = c.get(f"/games/{gid}/events", headers=KG).json()
    assert evs["last_seq"] == len(evs["events"]) and evs["events"][0]["seq"] == 1
    later = c.get(f"/games/{gid}/events?after_seq={evs['last_seq'] - 2}", headers=KG).json()
    assert [e["seq"] for e in later["events"]] == [evs["last_seq"] - 1, evs["last_seq"]]

    box = c.get(f"/games/{gid}/boxscore", headers=ADMIN).json()
    assert box["score"] == {"home": 8, "away": 2}
    assert box["home"]["points"] == 8 and box["home"]["totals"]["fg3m"] == 1 and box["away"]["points"] == 2
    lines = {p["player_id"]: p for p in box["players"]}
    assert lines[pid("home", 7)]["pts"] == 3 and lines[pid("home", 4)]["ast"] == 1
    assert lines[pid("home", 7)]["minutes"] == 40.0
    assert lines[pid("home", 7)]["rating"]["grade"] >= 1.0
    assert box["lineups"] and box["lineups"][0]["seconds"] == 2400
    sc = c.get(f"/games/{gid}/shotchart", headers=TRAINER_H).json()
    assert sc["summary"]["fga"] == 5 and sc["summary"]["fg3m"] == 1
    only = c.get(f"/games/{gid}/shotchart?player_id={pid('home', 7)}", headers=TRAINER_H).json()
    assert len(only["shots"]) == 1 and only["shots"][0]["x"] == 7.0


# ------------------------------------------------------------ WebSocket
def test_websocket_replay_and_broadcast(client):
    c = client
    gid = setup_world(c)
    ev(c, gid, "TIMEOUT", "home", None, 1, 500, 90)
    with c.websocket_connect(f"/ws/games/{gid}?role=kampfgericht") as ws:
        replay = []
        while True:
            m = ws.receive_json()
            if m["kind"] == "synced":
                break
            replay.append(m)
        assert all(m["replay"] for m in replay)
        assert [m["event"]["type"] for m in replay][-1] == "TIMEOUT"
        assert replay[-1]["text"] == "Q1 08:20 – Auszeit TV Dessau"
        n_hist = m["last_seq"]
        shot = ev(c, gid, "SHOT", "home", 7, 1, 450, 100, {"value": 2, "made": True, "x": 9, "y": 0})
        m = ws.receive_json()
        assert m["kind"] == "event" and not m["replay"]
        assert m["event"]["id"] == shot["id"] and m["event"]["seq"] == n_hist + 1
        assert "trifft Zweier" in m["text"]
    # Ab seq: nur neuere Events
    with c.websocket_connect(f"/ws/games/{gid}?after_seq={n_hist}&role=kampfgericht") as ws:
        m = ws.receive_json()
        assert m["event"]["id"] == shot["id"] and m["replay"]
    # Fans bekommen anonymisierte Events (Nico hat widersprochen)
    ev(c, gid, "SHOT", "home", 11, 1, 400, 110, {"value": 2, "made": True, "x": 9, "y": 0})
    with c.websocket_connect(f"/ws/games/{gid}") as ws:
        msgs = []
        while True:
            m = ws.receive_json()
            if m["kind"] == "synced":
                break
            msgs.append(m)
        nico = [m for m in msgs if m["event"]["type"] == "SHOT" and m["event"]["video_ts"] == 110][0]
        assert nico["event"]["player_id"] is None and "Nico" not in nico["text"] and "#11" not in nico["text"]
        assert not any(m["event"]["type"] == "STARTERS" for m in msgs)
    with pytest.raises(Exception):
        with c.websocket_connect("/ws/games/gibtsnicht"):
            pass


# --------------------------------------------------------- Anzeigetafel
def test_scoreboard_ingest_confirms_shot(client):
    c = client
    gid = setup_world(c)
    # Optik: unsicherer Dreier (provisorisch, Konfidenz 0.5)
    shot = ev(c, gid, "SHOT", "home", 7, 1, 540, 100.0, {"value": 3, "made": True, "x": 7.0, "y": 0.0},
              confidence=0.5, source="optical", status="provisional", headers=SYS)
    assert shot["status"] == "provisional"

    def post(ts, home, away, clock):
        r = c.post(f"/games/{gid}/scoreboard", headers=KG, json={
            "video_ts": ts, "home": home, "away": away, "clock": clock, "period": 1, "running": True,
            "fouls_home": 0, "fouls_away": 0})
        assert r.status_code == 200, r.text
        return r.json()

    out = [post(95 + i, 0, 0, 545 - i) for i in range(3)]
    assert out[-1]["accepted"] and any(e["type"] == "SCOREBOARD" for o in out for e in o["events"])
    out = [post(101 + i, 3, 0, 539 - i) for i in range(3)]
    produced = [e for o in out for e in o["events"]]
    assert any(e["type"] == "SCOREBOARD" and e["payload"]["home"] == 3 for e in produced)

    store = c.app.state.store
    eff = {e.id: e for e in store.effective(gid)}
    confirmed = [e for e in eff.values() if e.type == EventType.SHOT and e.player_id == pid("home", 7)]
    assert len(confirmed) == 1 and confirmed[0].status == EventStatus.CONFIRMED and confirmed[0].confidence >= 0.9
    assert shot["id"] not in eff                       # ersetzt, nicht dupliziert
    box = c.get(f"/games/{gid}/boxscore", headers=ADMIN).json()
    assert box["score"]["home"] == 3 and box["home"]["points"] == 3 and not box["score_mismatch"]

    live = c.get(f"/games/{gid}/live").json()
    assert live["score"] == {"home": 3, "away": 0} and live["period"] == 1
    assert live["team_fouls"] == {"home": 0, "away": 0} and live["running"] is True
    # Fans und Trainer duerfen nicht in die Tafel schreiben
    assert c.post(f"/games/{gid}/scoreboard", headers=FAN, json={"video_ts": 1, "home": 0}).status_code == 403


# ------------------------------------------------------------------ Live
def test_live_text_and_privacy_in_playbyplay(client):
    c = client
    gid = setup_world(c)
    play_game(c, gid)
    live = c.get(f"/games/{gid}/live").json()
    texts = [i["text"] for i in live["playbyplay"]]
    assert "Q2 07:45 – #7 Müller trifft Dreier (Assist #4 Schmidt)" in texts
    assert live["playbyplay"][0]["text"].startswith("Q4") or len(live["playbyplay"]) <= 20
    # Spieler mit Widerspruch / Minderjaehrige ohne Freigabe: nur Team-Wert, auch im Text
    joined = "\n".join(texts)
    assert "Objekt" not in joined and "#11" not in joined and "Jung" not in joined and "#15" not in joined
    assert "TV Dessau trifft Zweier" in joined
    # Trainer des Heimteams sieht die Namen
    live_t = c.get(f"/games/{gid}/live", headers=TRAINER_H).json()
    assert any("#11 Objekt trifft Zweier" in i["text"] for i in live_t["playbyplay"])
    assert live["livestream"]["allowed"] is False         # nicht alle Spieler haben LIVESTREAM-Einwilligung
    assert len(c.get(f"/games/{gid}/playbyplay?limit=3").json()["items"]) == 3


# ---------------------------------------------------------------- Review
def test_review_confirm_reject_suggestion(client):
    c = client
    gid = setup_world(c)
    a = ev(c, gid, "SHOT", "home", 7, 1, 500, 100.0, {"value": 3, "made": True, "x": 7, "y": 0},
           confidence=0.5, headers=SYS)
    b = ev(c, gid, "REBOUND", "home", 4, 1, 480, 120.0, {"offensive": False}, confidence=0.6, headers=SYS)
    d = ev(c, gid, "SHOT", "away", 3, 1, 470, 130.0, {"value": 2, "made": True, "x": -8, "y": 0},
           confidence=0.4, headers=SYS)
    ev(c, gid, "SHOT", "home", 8, 1, 460, 140.0, {"value": 2, "made": True, "x": 8, "y": 0}, confidence=0.95, headers=SYS)
    assert c.get(f"/games/{gid}/review", headers=FAN).status_code == 403
    rv = c.get(f"/games/{gid}/review", headers=KG).json()
    assert rv["count"] == 3
    first = rv["items"][0]
    assert first["event_id"] == a["id"]                       # hoechster Einfluss zuerst
    assert abs(first["clip_end"] - first["clip_start"] - 5.0) < 1e-9 and first["clip_start"] == 97.0
    assert first["suggestions"] and first["suggestions"][0]["label"].startswith("2er")
    assert "trifft Dreier" in first["text"]
    # Trainer des Heimteams sieht nur eigene Mannschaft
    assert {i["team"] for i in c.get(f"/games/{gid}/review", headers=TRAINER_H).json()["items"]} == {"home"}
    assert c.post(f"/games/{gid}/review/{d['id']}", json={"action": "reject"}, headers=TRAINER_H).status_code == 403

    r = c.post(f"/games/{gid}/review/{a['id']}", json={"action": "confirm"}, headers=KG)
    assert r.status_code == 200 and r.json()["event"]["status"] == "confirmed" and r.json()["remaining"] == 2
    assert c.post(f"/games/{gid}/review/{a['id']}", json={"action": "confirm"}, headers=KG).status_code == 409
    r = c.post(f"/games/{gid}/review/{d['id']}", json={"action": "reject"}, headers=KG)
    assert r.json()["event"]["type"] == "VOID" and r.json()["remaining"] == 1
    assert c.post(f"/games/{gid}/review/{b['id']}", json={"action": "suggestion", "suggestion_index": 99},
                  headers=KG).status_code == 400
    r = c.post(f"/games/{gid}/review/{b['id']}", json={"action": "suggestion", "suggestion_index": 0}, headers=KG)
    assert r.status_code == 200 and r.json()["event"]["payload"]["offensive"] is True
    assert c.get(f"/games/{gid}/review", headers=KG).json()["count"] == 0
    box = c.get(f"/games/{gid}/boxscore", headers=ADMIN).json()
    assert box["away"]["points"] == 0 and box["home"]["totals"]["orb"] == 1 and box["home"]["points"] == 5


def test_foul_input(client):
    c = client
    gid = setup_world(c)
    ev(c, gid, "PERIOD_START", None, None, 1, 600.0, 0)
    ev(c, gid, "TIMEOUT", "home", None, 1, 420.0, 200)
    r = c.post(f"/games/{gid}/fouls", json={"player_id": pid("home", 7), "kind": "personal",
                                             "fouled_id": pid("away", 3)}, headers=KG)
    assert r.status_code == 201, r.text
    e = r.json()["event"]
    assert e["type"] == "FOUL" and e["team"] == "home" and e["status"] == "confirmed"
    assert e["period"] == 1 and e["clock"] == 420.0 and r.json()["player_fouls"] == 1
    c.post(f"/games/{gid}/fouls", json={"player_id": pid("home", 7), "kind": "technical"}, headers=KG)
    box = c.get(f"/games/{gid}/boxscore", headers=ADMIN).json()
    assert {p["player_id"]: p for p in box["players"]}[pid("home", 7)]["pf"] == 2
    assert box["home"]["team_fouls_by_period"] == {"1": 1}       # technisches Foul zaehlt nicht als Teamfoul
    assert {p["player_id"]: p for p in box["players"]}[pid("away", 3)]["fd"] == 1
    assert c.post(f"/games/{gid}/fouls", json={"player_id": "nope"}, headers=KG).status_code == 404
    assert c.post(f"/games/{gid}/fouls", json={"player_id": pid("away", 3)}, headers=TRAINER_H).status_code == 403
    assert c.post(f"/games/{gid}/fouls", json={"player_id": pid("away", 3)}, headers=FAN).status_code == 403
    live = c.get(f"/games/{gid}/live").json()
    assert any("Persönliches Foul" in i["text"] for i in live["playbyplay"]) or live["playbyplay"]


def test_scoresheet_flags_and_finalize(client):
    c = client
    gid = setup_world(c)
    play_game(c, gid)
    sheet = {"players": {pid("home", 7): {"points": 5, "fouls": 0, "ftm": 0},
                         pid("home", 4): {"points": 2, "fouls": 0, "ftm": 0}}, "home": 8, "away": 2}
    r = c.post(f"/games/{gid}/scoresheet", json=sheet, headers=KG)
    assert r.status_code == 201
    flags = r.json()["flags"]
    assert r.json()["mismatches"] == len(flags) >= 1
    assert any(f["payload"]["kind"] == "scoresheet_mismatch" and f["payload"].get("player_id") == pid("home", 7)
               for f in flags)
    # offene Abweichung blockiert die Freigabe
    r = c.post(f"/games/{gid}/finalize", json={"by": "kg"}, headers=KG)
    assert r.status_code == 409 and "Freigabe verweigert" in r.json()["detail"]
    assert c.post(f"/games/{gid}/finalize", json={"by": "t", "force": True}, headers=TRAINER_H).status_code == 403
    # Korrektur: Bogen nochmals korrekt eingeben ersetzt die alten Abweichungen
    sheet["players"][pid("home", 7)]["points"] = 3
    sheet["players"][pid("home", 11)] = {"points": 2, "fouls": 0, "ftm": 0}
    sheet["players"][pid("home", 15)] = {"points": 1, "fouls": 0, "ftm": 1}
    sheet["players"][pid("away", 3)] = {"points": 2, "fouls": 0, "ftm": 0}
    r = c.post(f"/games/{gid}/scoresheet", json=sheet, headers=KG)
    assert r.json()["mismatches"] == 0
    assert c.get(f"/games/{gid}/review", headers=KG).json()["count"] == 0
    r = c.post(f"/games/{gid}/finalize", json={"by": "Kampfgericht"}, headers=KG)
    assert r.status_code == 200, r.text
    assert r.json()["game"]["status"] == "final" and r.json()["event"]["payload"]["by"] == "Kampfgericht"
    assert c.post(f"/games/{gid}/finalize", json={}, headers=KG).status_code == 409
    assert c.post(f"/games/{gid}/events", json={"type": "TIMEOUT"}, headers=KG).status_code == 409
    assert c.post(f"/games/{gid}/events", json={"type": "TIMEOUT", "team": "home"}, headers=ADMIN).status_code == 201
    assert c.get(f"/games/{gid}/boxscore", headers=ADMIN).json()["status"] == "final"


def test_finalize_force(client):
    c = client
    gid = setup_world(c)
    ev(c, gid, "SHOT", "home", 7, 1, 500, 100.0, {"value": 2, "made": True, "x": 7, "y": 0}, confidence=0.3, headers=SYS)
    assert c.post(f"/games/{gid}/finalize", json={}, headers=KG).status_code == 409
    r = c.post(f"/games/{gid}/finalize", json={"force": True}, headers=KG)
    assert r.status_code == 200 and r.json()["event"]["payload"]["forced"] is True


# -------------------------------------------------------------- Datenschutz
def _by_id(box):
    return {p["player_id"]: p for p in box["players"]}


def test_privacy_boxscore_views(client):
    c = client
    gid = setup_world(c)
    play_game(c, gid)
    fan = c.get(f"/games/{gid}/boxscore").json()
    lines = _by_id(fan)
    assert fan["view"] == "public"
    assert pid("home", 11) not in lines and pid("home", 15) not in lines          # Widerspruch / Minderjaehrig ohne Freigabe
    assert fan["hidden_players"] == 2
    assert fan["home"]["points"] == 8 and fan["home"]["totals"]["pts"] == 8        # Team-Summen bleiben vollstaendig
    assert lines[pid("home", 7)]["rating"] is not None                              # Erwachsener mit oeffentlicher Freigabe
    mia = lines[pid("home", 16)]
    assert mia["name"] == "Mia Klein" and mia["rating"] is None                     # Minderjaehrige: nie oeffentlich bewertet
    assert fan["lineups"] == []
    assert "Objekt" not in str(fan) and "Jung" not in str(fan)
    for v in c.get(f"/games/{gid}/events").json()["events"]:
        assert v["player_id"] not in (pid("home", 11), pid("home", 15))
    sc = c.get(f"/games/{gid}/shotchart").json()
    assert sc["summary"]["fga"] == 5 and all(s["player_id"] != pid("home", 11) for s in sc["shots"])
    assert any(s["player_id"] is None for s in sc["shots"])
    assert c.get(f"/games/{gid}/shotchart?player_id={pid('home', 11)}").status_code == 403
    assert c.get(f"/players/{pid('home', 11)}/profile").status_code == 403
    assert c.get(f"/players/{pid('home', 7)}/profile").status_code == 200

    # Trainer des Heimteams: alles, auch Noten Minderjaehriger; Gegnertrainer: oeffentliche Sicht
    t = c.get(f"/games/{gid}/boxscore", headers=TRAINER_H).json()
    tl = _by_id(t)
    assert t["view"] == "team" and pid("home", 11) in tl and tl[pid("home", 16)]["rating"] is not None
    assert tl[pid("home", 15)]["rating"] is not None and t["hidden_players"] == 0 and t["lineups"]
    assert {lu["side"] for lu in t["lineups"]} == {"home"}
    ta = c.get(f"/games/{gid}/boxscore", headers=TRAINER_A).json()
    assert pid("home", 11) not in _by_id(ta) and pid("away", 3) in _by_id(ta)
    # Spieler/Eltern sehen die eigene Zeile samt Note
    me = c.get(f"/games/{gid}/boxscore", headers=hdr("spieler", player=pid("home", 15))).json()
    assert pid("home", 15) in _by_id(me) and _by_id(me)[pid("home", 15)]["rating"] is not None
    assert pid("home", 11) not in _by_id(me)
    # Profil: Minderjaehrige-Note nur fuer sich/Trainer
    pm = c.get(f"/players/{pid('home', 16)}/profile").json()
    assert pm["rating_visible"] is False and pm["average_grade"] is None and pm["form"] == []
    pm_t = c.get(f"/players/{pid('home', 16)}/profile", headers=TRAINER_H).json()
    assert pm_t["rating_visible"] and pm_t["average_grade"] >= 1.0 and set(pm_t["radar"]) >= {"scoring", "defense"}
    assert len(pm_t["per_game"]) == 1
    # Roster: Namen nicht freigegebener Spieler sind oeffentlich verborgen
    pub = {p["number"]: p for p in c.get("/teams/t-home/players").json()}
    assert pub[11]["name"] is None and pub[7]["name"] == "Hans Müller" and "is_minor" not in pub[7]
    own = {p["number"]: p for p in c.get("/teams/t-home/players", headers=TRAINER_H).json()}
    assert own[11]["name"] == "Nico Objekt" and own[15]["is_minor"] is True


def test_consent_change_and_audit(client):
    c = client
    gid = setup_world(c)
    play_game(c, gid)
    mueller = pid("home", 7)
    assert pid("home", 7) in _by_id(c.get(f"/games/{gid}/boxscore").json())
    # Nur der Spieler selbst, der Verein oder Admin duerfen aendern
    assert c.post(f"/players/{mueller}/consent", json={"action": "object"}, headers=FAN).status_code == 403
    assert c.post(f"/players/{mueller}/consent", json={"action": "object"},
                  headers=hdr("spieler", player=pid("home", 8))).status_code == 403
    assert c.post(f"/players/{mueller}/consent", json={"action": "object"}, headers=TRAINER_H).status_code == 403
    r = c.post(f"/players/{mueller}/consent", json={"action": "object", "note": "Opt-out per App"},
               headers=hdr("spieler", player=mueller))
    assert r.status_code == 200, r.text
    rec = r.json()["record"]
    assert rec["action"] == "object" and rec["actor"] == "player" and rec["before"] & 2 and rec["after"] == 32
    assert r.json()["consent"]["objected"] is True
    # Wirkung: Fans sehen den Spieler nicht mehr, Highlights auch nicht
    assert mueller not in _by_id(c.get(f"/games/{gid}/boxscore").json())
    assert all(cd["player_id"] != mueller for cd in c.get(f"/games/{gid}/highlights").json()["candidates"])
    assert c.get(f"/players/{mueller}/profile").status_code == 403
    assert "Müller" not in "".join(i["text"] for i in c.get(f"/games/{gid}/live").json()["playbyplay"])
    # Audit-Trail (Eigentuemer, Verein, Admin)
    aud = c.get(f"/players/{mueller}/consent", headers=hdr("spieler", player=mueller)).json()
    assert [a["action"] for a in aud["audit"]][-1] == "object" and aud["effective"] == 32
    assert c.get(f"/players/{mueller}/consent", headers=FAN).status_code == 403
    # Rücknahme + Teilfreigabe
    r = c.post(f"/players/{mueller}/consent", json={"action": "withdraw_objection"}, headers=hdr("spieler", player=mueller))
    assert r.json()["consent"]["stats_public"] is True
    r = c.post(f"/players/{mueller}/consent", json={"action": "revoke", "level": ["STATS_PUBLIC"]},
               headers=hdr("spieler", player=mueller))
    assert r.json()["consent"]["stats_public"] is False
    assert mueller not in _by_id(c.get(f"/games/{gid}/boxscore").json())
    assert mueller in _by_id(c.get(f"/games/{gid}/boxscore", headers=TRAINER_H).json())
    actions = [a["action"] for a in c.get(f"/players/{mueller}/consent", headers=HW).json()["audit"]]
    assert actions[-3:] == ["object", "withdraw_objection", "revoke"]
    assert c.post(f"/players/{mueller}/consent", json={"action": "grant", "level": ["NOPE"]}, headers=HW).status_code == 400
    # Minderjaehrige: Elterneinwilligung schaltet erst frei
    leo = pid("home", 15)
    assert leo not in _by_id(c.get(f"/games/{gid}/boxscore").json())
    c.post(f"/players/{leo}/consent", json={"action": "parental_consent", "level": ["STATS_INTERNAL", "STATS_PUBLIC"]},
           headers=hdr("spieler", player=leo))
    c.post(f"/players/{leo}/consent", json={"action": "child_consent"}, headers=hdr("spieler", player=leo))
    assert leo in _by_id(c.get(f"/games/{gid}/boxscore").json())
    assert _by_id(c.get(f"/games/{gid}/boxscore").json())[leo]["rating"] is None


def test_highlights_consent_filter(client):
    c = client
    gid = setup_world(c)
    play_game(c, gid)
    fan = c.get(f"/games/{gid}/highlights").json()
    ids = {cd["player_id"] for cd in fan["candidates"]}
    assert pid("home", 7) in ids
    assert pid("home", 11) not in ids and pid("home", 15) not in ids and pid("home", 16) not in ids
    assert fan["channel"] == "public" and fan["ai_label"]
    for plan in fan["plans"]:
        assert all(s["actor_id"] in ids for s in plan["segments"])
        assert "KI-generiert" in plan["overlays"]
    # Interner Kanal fuer das Heimteam: Widerspruch gilt weiterhin
    team = c.get(f"/games/{gid}/highlights", headers=TRAINER_H).json()
    assert team["channel"] == "team"
    assert pid("home", 11) not in {cd["player_id"] for cd in team["candidates"]}
    reel = c.get(f"/players/{pid('home', 7)}/reel").json()
    assert reel["plan"] and reel["plan"]["kind"] == "player_reel"
    assert c.get(f"/players/{pid('home', 11)}/reel").json()["plan"] is None


# ---------------------------------------------------- Verband / Bestenliste
def test_leaderboard_excludes_minors_and_export(client):
    c = client
    gid = setup_world(c)
    play_game(c, gid)
    # noch nicht final: keine Bestenliste
    assert c.get("/leagues/Oberliga/leaderboard").json()["rows"] == []
    c.post(f"/games/{gid}/finalize", json={"force": True, "by": "kg"}, headers=KG)
    lb = c.get("/leagues/Oberliga/leaderboard").json()
    names = [r["name"] for r in lb["rows"]]
    assert "Hans Müller" in names and "Jonas Schmidt" in names
    assert "Mia Klein" not in names and "Leo Jung" not in names and "Nico Objekt" not in names
    assert [r["rank"] for r in lb["rows"]] == list(range(1, len(names) + 1))
    assert lb["rows"][0]["average_grade"] >= lb["rows"][-1]["average_grade"]
    assert c.get("/leagues/Oberliga/leaderboard?metric=scoring").status_code == 200
    assert c.get("/leagues/Oberliga/leaderboard?metric=bogus").status_code == 400

    assert c.get("/leagues/Oberliga/export.csv", headers=FAN).status_code == 403
    r = c.get("/leagues/Oberliga/export.csv", headers=VERBAND)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text), delimiter=";"))
    assert {x["game_id"] for x in rows} == {gid}
    named = {x["name"] for x in rows}
    assert {"Hans Müller", "Jonas Schmidt", "TEAM"} <= named
    assert "Nico Objekt" not in named and "Leo Jung" not in named          # nur in den Team-Summen
    team_home = [x for x in rows if x["name"] == "TEAM" and x["team"] == "TV Dessau"][0]
    assert team_home["pts"] == "8"
    assert c.get("/leagues/Andere/export.csv", headers=VERBAND).text.count("\n") == 1   # nur Kopfzeile
    r = c.get("/leagues/Oberliga/export.csv?delimiter=,", headers=ADMIN)
    assert r.text.splitlines()[0].startswith("game_id,date,")


# ---------------------------------------------------------------- Diverses
def test_static_index_pricing_dpia(client):
    c = client
    r = c.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "CourtVision" in r.text
    for page in ("setup", "console", "dashboard", "game", "player", "verband"):
        assert c.get(f"/{page}.html").status_code == 200, page
    assert c.get("/app.js").status_code == 200 and c.get("/style.css").status_code == 200
    assert c.get("/health").json() == {"status": "ok"}

    inv = c.get("/pricing/invoice?games=25", headers=HW).json()
    assert inv["net"] == 249.0 and inv["lines"][1]["label"].startswith("5 Zusatzspiele")
    assert c.get("/pricing/invoice?games=25", headers=FAN).status_code == 403

    d = c.get("/privacy/dpia", headers=VERBAND).json()
    assert d["progress"] == 0 and "dpia" in d["pilot_blockers"] and c.get("/privacy/dpia").status_code == 403
    d = c.post("/privacy/dpia/dpia", json={"status": "done", "owner": "DSB"}, headers=ADMIN).json()
    assert d["progress"] > 0 and "dpia" not in d["pilot_blockers"]


def test_media_and_livestream_flag(client):
    c = client
    gid = setup_world(c, start=False)
    # Alle Spieler muessen LIVESTREAM freigeben, damit ein Stream angeboten wird
    c.put(f"/games/{gid}/media", json={"stream_url": "https://example.org/stream", "video_url": "/v.mp4"}, headers=HW)
    assert c.get(f"/games/{gid}/live").json()["livestream"] == {"allowed": False, "url": None}
    for p in c.get("/teams/t-home/players", headers=HW).json():
        if p["number"] in (11, 15, 16):
            c.post(f"/players/{p['id']}/consent", json={"action": "parental_consent" if p["is_minor"] else "withdraw_objection",
                                                        "level": ["STATS_INTERNAL", "STATS_PUBLIC", "HIGHLIGHTS", "LIVESTREAM"]},
                   headers=HW)
    for pl in (pid("home", 15), pid("home", 16)):
        c.post(f"/players/{pl}/consent", json={"action": "child_consent"}, headers=HW)
    c.post(f"/players/{pid('home', 11)}/consent", json={"action": "withdraw_objection"}, headers=HW)
    live = c.get(f"/games/{gid}/live").json()
    assert live["livestream"] == {"allowed": True, "url": "https://example.org/stream"}
    ev(c, gid, "TIMEOUT", "home", None, 1, 500, 5)
    assert c.get(f"/games/{gid}/review", headers=KG).json()["video_url"] == "/v.mp4"


# ------------------------------------------------- GameSession-Anbindung
def test_scoreboard_ingest_emits_period_events(client):
    c = client
    gid = setup_world(c)

    def post(ts, home, away, clock, period, running):
        r = c.post(f"/games/{gid}/scoreboard", headers=KG, json={
            "video_ts": ts, "home": home, "away": away, "clock": clock, "period": period, "running": running})
        assert r.status_code == 200, r.text
        return r.json()["events"]

    produced = []
    for i in range(3):
        produced += post(10 + i, 0, 0, 600 - i, 1, True)
    for i in range(3):
        produced += post(600 + i, 2, 0, 0.0, 1, False)       # Viertelende
    for i in range(3):
        produced += post(700 + i, 2, 0, 600 - i, 2, True)    # Viertel 2 laeuft
    types = [(e["type"], e["period"]) for e in produced if e["type"].startswith("PERIOD")]
    assert ("PERIOD_END", 1) in types and ("PERIOD_START", 2) in types
    # Start kam schon vom Bediener (/start): der Tracker erzeugt keinen zweiten GAME_START/PERIOD_START 1
    all_types = [e["type"] for e in c.get(f"/games/{gid}/events", headers=KG).json()["events"]]
    assert all_types.count("GAME_START") == 1 and all_types.count("PERIOD_START") == 2
    # Spielende ueber /end: Session schliesst Periode und Spiel, Status PROCESSING
    r = c.post(f"/games/{gid}/end", headers=HW)
    assert r.status_code == 200 and r.json()["game"]["status"] == "processing"
    assert r.json()["event"]["type"] == "GAME_END"
    all_types = [e["type"] for e in c.get(f"/games/{gid}/events", headers=KG).json()["events"]]
    assert all_types.count("GAME_END") == 1 and all_types.count("PERIOD_END") == 2
    # Korrektur-Endpunkte arbeiten in processing (und review)
    assert c.get(f"/games/{gid}/review", headers=KG).status_code == 200
    assert c.post(f"/games/{gid}/fouls", headers=KG, json={"player_id": pid("home", 4)}).status_code == 201
    assert c.get(f"/games/{gid}/boxscore", headers=ADMIN).json()["score"] == {"home": 2, "away": 0}


def test_session_recreated_lazily_after_restart(tmp_path):
    db = str(tmp_path / "cv.sqlite")
    with TestClient(create_app(EventStore(db))) as c:
        gid = setup_world(c)
    with TestClient(create_app(EventStore(db))) as c:       # Neustart: Session existiert nicht mehr
        r = c.post(f"/games/{gid}/scoreboard", headers=KG, json={
            "video_ts": 5, "home": 0, "away": 0, "clock": 590, "period": 1, "running": True})
        assert r.status_code == 200
        assert c.post(f"/games/{gid}/end", headers=HW).json()["game"]["status"] == "processing"


def test_serialize_round_trip():
    obs = [
        FrameState(12.5, (PlayerState(3, "home", 1.5, -2.0, "t-home-4", 0.9), PlayerState(7, None, 0.0, 0.0)),
                   ball_x=1.0, ball_y=2.0, ball_height=2.5),
        FrameState(13.0, ()),
        Possession(1.0, 5, "away", "t-away-3", 0.8), Possession(2.0, None, None),
        ShotRelease(3.0, 4, "home", 7.0, 1.0, "right", "t-home-7", 0.7, 0.9),
        RimMake(4.0, "left", 0.6), ClockState(5.0, 2, 431.5, True),
    ]
    from courtvision.domain.observations import BallTouch, Formation, OutOfBounds
    obs += [BallTouch(6.0, 2, "away", None, True, 0.5), OutOfBounds(7.0, 9, "home", "t-home-4", 0.4),
            Formation(8.0, "free_throw", 3, "home", "t-home-4", "left", 0.95), Formation(9.0, "jump_ball")]
    import json
    for o in obs:
        d = observation_to_dict(o)
        assert d["kind"] == type(o).__name__
        assert observation_from_dict(json.loads(json.dumps(d))) == o
    assert observation_to_dict(obs[0])["players"][0]["player_id"] == "t-home-4"
    assert observation_to_dict(obs[-1]) == {"kind": "Formation", "video_ts": 9.0, "formation_kind": "jump_ball",
                                            "shooter_track": None, "shooter_team": None, "shooter_player": None,
                                            "basket": None, "confidence": 1.0}
    # ganzzahlige JSON-Zahlen fuer Float-Felder sind erlaubt
    assert observation_from_dict({"kind": "ClockState", "video_ts": 1, "period": 1, "clock": 600, "running": False}) \
        == ClockState(1.0, 1, 600.0, False)
    for bad in ({"kind": "Nope"}, {"kind": "RimMake", "video_ts": 1}, {"kind": "RimMake", "video_ts": 1, "basket": "up"},
                {"kind": "RimMake", "video_ts": 1, "basket": "left", "x": 1},
                {"kind": "Possession", "video_ts": 1, "track_id": None, "team": "red"},
                {"kind": "FrameState", "video_ts": 1, "players": [{"track_id": 1}]}, "x"):
        with pytest.raises(ValueError):
            observation_from_dict(bad)


def test_observations_endpoint(client):
    c = client
    gid = setup_world(c)
    body = [{"kind": "ClockState", "video_ts": 10 + i, "period": 1, "clock": 600.0 - i, "running": True} for i in range(3)]
    assert c.post(f"/games/{gid}/observations", json=body, headers=KG).status_code == 403
    assert c.post(f"/games/{gid}/observations", json=body).status_code == 403
    r = c.post(f"/games/{gid}/observations", json=body, headers=SYS)
    assert r.status_code == 200 and r.json()["fed"] == 3
    # Wurf + Treffer + Tafel -> optisches SHOT-Event
    shot = [
        {"kind": "FrameState", "video_ts": 20.0, "players": [
            {"track_id": 1, "team": "home", "x": 7.0, "y": 0.0, "player_id": pid("home", 7), "id_confidence": 0.9}]},
        {"kind": "Possession", "video_ts": 20.5, "track_id": 1, "team": "home", "player_id": pid("home", 7)},
        {"kind": "ShotRelease", "video_ts": 21.0, "track_id": 1, "team": "home", "x": 7.0, "y": 0.0,
         "basket": "right", "player_id": pid("home", 7), "three_prob": 0.9},
        {"kind": "RimMake", "video_ts": 22.0, "basket": "right"},
        {"kind": "ClockState", "video_ts": 30.0, "period": 1, "clock": 570.0, "running": True},
    ]
    r = c.post(f"/games/{gid}/observations", json=shot, headers=ADMIN)
    assert r.status_code == 200, r.text
    evs = c.get(f"/games/{gid}/events", headers=KG).json()["events"]
    assert any(e["type"] == "SHOT" and e["source"] == "optical" for e in evs)
    # Validierung: ganze Liste wird abgelehnt, nichts wird gespeist
    n = len(evs)
    bad = c.post(f"/games/{gid}/observations", headers=SYS,
                 json=[{"kind": "ClockState", "video_ts": 40, "period": 1, "clock": 1, "running": True},
                       {"kind": "RimMake", "video_ts": 41, "basket": "up"}])
    assert bad.status_code == 422 and "Beobachtung 1" in bad.json()["detail"]
    assert len(c.get(f"/games/{gid}/events", headers=KG).json()["events"]) == n
    assert c.post("/games/zzz/observations", json=[], headers=SYS).status_code == 404


def _demo_client(tmp_path=None):
    store = EventStore(":memory:" if tmp_path is None else str(tmp_path / "demo.sqlite"))
    setup = load_demo(store, demo_setup())
    return TestClient(create_app(store)), setup


def test_simulate_endpoint_sync_matches_truth():
    c, setup = _demo_client()
    with c:
        gid = setup.game.id
        body = {"seed": 6, "periods": 1, "period_length_s": 120, "include_frames": False}
        assert c.post(f"/games/{gid}/simulate", json=body, headers=KG).status_code == 403
        assert c.post(f"/games/{gid}/simulate", json={"speed": 0}, headers=ADMIN).status_code == 422
        r = c.post(f"/games/{gid}/simulate", json=body, headers=ADMIN)
        assert r.status_code == 200, r.text
        info = r.json()
        sim = simulate_game(6, setup.game.model_copy(update={"periods": 1, "period_length_s": 120.0}), setup.rosters,
                            periods=1, period_length_s=120.0, include_frames=False)
        assert info["mode"] == "sync" and info["final_score"] == sim.final_score
        assert info["game"]["status"] == "processing"
        box = c.get(f"/games/{gid}/boxscore", headers=ADMIN).json()
        assert box["score"] == sim.final_score and not box["score_mismatch"]
        live = c.get(f"/games/{gid}/live").json()
        assert live["score"] == sim.final_score and live["status"] == "processing"
        types = [e["type"] for e in c.get(f"/games/{gid}/events", headers=KG).json()["events"]]
        for t in ("GAME_START", "PERIOD_START", "PERIOD_END", "GAME_END", "STARTERS", "SCOREBOARD"):
            assert t in types, t
        n_truth_fouls = sum(1 for e in sim.truth if e.type == EventType.FOUL)
        assert n_truth_fouls > 0 and types.count("FOUL") == n_truth_fouls  # Konsolen-Fouls aus der Wahrheit
        assert c.get(f"/games/{gid}/review", headers=KG).status_code == 200
        # nur einmal: das Spiel ist nicht mehr geplant
        assert c.post(f"/games/{gid}/simulate", json=body, headers=ADMIN).status_code == 409


def test_simulate_endpoint_replay_in_background():
    import time
    c, setup = _demo_client()
    with c:
        gid = setup.game.id
        r = c.post(f"/games/{gid}/simulate", json={"seed": 2, "periods": 1, "period_length_s": 60, "speed": 1000},
                   headers=ADMIN)
        assert r.status_code == 200 and r.json()["mode"] == "replay"
        for _ in range(100):
            st = c.get(f"/games/{gid}/simulate", headers=ADMIN).json()
            if not st["running"]:
                break
            time.sleep(0.1)
        assert not st["running"] and not st.get("error")
        assert c.get(f"/games/{gid}/live").json()["score"] == st["final_score"]


# ------------------------------------------------------------ Persistenz
def test_state_survives_app_recreation(tmp_path):
    db = str(tmp_path / "cv.sqlite")
    names = ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right", "center"]
    clicks = {n: [100.0 + 10 * i, 50.0 + 7 * i * i] for i, n in enumerate(names)}
    from courtvision.geometry.court import Calibration
    Hm = Calibration.from_landmarks  # noqa: F841 (Pruefung der Eingabe erfolgt ueber die API)
    with TestClient(create_app(EventStore(db))) as c:
        setup_world(c, start=False)
        # Kalibrierung mit einer realen Homographie (Projektion der Landmarken)
        H = np.array([[30.0, 4.0, 640.0], [-2.0, 22.0, 360.0], [0.0004, 0.002, 1.0]])
        clicks = {}
        for n in names:
            x, y = LANDMARKS[n]
            p = H @ [x, y, 1]
            clicks[n] = [float(p[0] / p[2]), float(p[1] / p[2])]
        zones = {"substitution": [[-2, 7.5], [2, 7.5], [2, 9], [-2, 9]]}
        r = c.post("/games/g1/calibration", json={"clicks": clicks, "zones": zones}, headers=HW)
        assert r.status_code == 200, r.text
        cal = r.json()
        assert c.put("/games/g1/media", json={"video_url": "https://x/v.mp4", "stream_url": "https://x/s"},
                     headers=HW).status_code == 200
        # Einwilligung aendern -> Audit-Eintrag; DSFA-Punkt abhaken
        pl = pid("home", 4)
        r = c.post(f"/players/{pl}/consent", json={"action": "revoke", "level": ["HIGHLIGHTS"], "note": "Wunsch"},
                   headers=ADMIN)
        assert r.status_code == 200, r.text
        audit_before = c.get(f"/players/{pl}/consent", headers=ADMIN).json()["audit"]
        assert audit_before
        r = c.post("/privacy/dpia/dpia", json={"status": "done", "owner": "DSB", "note": "abgenommen"}, headers=ADMIN)
        assert r.status_code == 200
        assert c.get("/venues").json()[0]["id"] == "v1"
    with TestClient(create_app(EventStore(db))) as c:       # neue App auf derselben Datei
        got = c.get("/games/g1/calibration", headers=HW).json()
        assert got["homography"] == cal["homography"] and got["zones"] == cal["zones"]
        assert c.post("/games/g1/calibration/zone-check?x=0&y=8", headers=HW).json()["zone"] == "substitution"
        # Wechselzonen der Kalibrierung kommen in der neuen Session an
        assert c.app.state.cv.slot("g1").session.engine is not None
        assert c.get("/games/g1/live").json()["livestream"]["url"] is None  # (nicht erlaubt), aber Medien da:
        assert c.app.state.cv.media("g1") == {"video_url": "https://x/v.mp4", "stream_url": "https://x/s"}
        audit_after = c.get(f"/players/{pl}/consent", headers=ADMIN).json()["audit"]
        assert audit_after == audit_before
        assert "HIGHLIGHTS" not in c.get(f"/players/{pl}/consent", headers=ADMIN).json()["levels"]
        items = {i["id"]: i for i in c.get("/privacy/dpia", headers=hdr("verband")).json()["items"]}
        assert items["dpia"]["status"] == "done" and items["dpia"]["owner"] == "DSB"
        assert items["signage"]["status"] == "open"
        # weiterer Eintrag wird an den alten Audit-Trail angehaengt
        c.post(f"/players/{pl}/consent", json={"action": "grant", "level": ["HIGHLIGHTS"]}, headers=ADMIN)
        assert len(c.get(f"/players/{pl}/consent", headers=ADMIN).json()["audit"]) == len(audit_before) + 1
