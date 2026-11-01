from datetime import datetime, timedelta

import numpy as np
import pytest

from courtvision.business.pricing import CostPerGame, PricingPlan, invoice, livestream_split, unit_economics
from courtvision.domain.events import (
    Event, EventSource, EventStatus, EventType, FoulKind, effective_events,
)
from courtvision.domain.models import Consent, Game, GameStatus, Player
from courtvision.highlights.camera import CameraConfig, virtual_camera
from courtvision.highlights.clips import (
    build_scenes, filter_publishable, live_clip, player_reel, recap, render_command, season_reel,
    vertical_clip,
)
from courtvision.highlights.overlay import build_overlay, overlay_text_lines
from courtvision.highlights.scoring import build_contexts, highlight_score, score_events
from courtvision.privacy.consent import (
    ConsentManager, can_show_stats_public, livestream_allowed, public_ratings_allowed, team_only_stats,
)
from courtvision.privacy.dpia_checklist import ItemStatus, default_checklist
from courtvision.privacy.retention import (
    Recording, due_for_deletion, purge_transient_biometrics, register_biometric_store,
)
from courtvision.qa.review import (
    ConsoleFoulInput, FinalizationError, accuracy_report, apply_review, correction_queue, event_f1,
    finalize, reconcile_scoresheet,
)
from courtvision.store.eventstore import EventStore

G = "g1"


def ev(type_, ts, team="home", pid=None, conf=1.0, status=EventStatus.PROVISIONAL, period=1, clock=500.0,
       **payload):
    return Event(game_id=G, type=type_, video_ts=ts, team=team, player_id=pid, confidence=conf,
                 status=status, period=period, clock=clock, payload=payload)


def shot(ts, pid="h1", value=2, made=True, **kw):
    return ev(EventType.SHOT, ts, pid=pid, value=value, made=made, x=0.0, y=0.0, **kw)


def roster():
    return {
        "home": [Player(id=f"h{i}", team_id="H", number=i, name=f"Home{i}") for i in range(1, 8)],
        "away": [Player(id=f"a{i}", team_id="A", number=i, name=f"Away{i}") for i in range(1, 8)],
    }


# ------------------------------------------------------------------ QA
def test_queue_sorted_by_impact_with_clip_and_suggestions():
    r = roster()
    starters = ev(EventType.STARTERS, 0, players=["h1", "h2", "h3", "h4", "h5"])
    foul = ev(EventType.FOUL, 10, pid="h2", conf=0.3, kind=FoulKind.PERSONAL, fouled_id=None)
    reb = ev(EventType.REBOUND, 20, pid="h3", conf=0.5, offensive=False)
    sub = ev(EventType.SUBSTITUTION, 30, conf=0.4, player_in="h6", player_out="h1",
             candidates=[{"player_in": "h7", "player_out": "h1"}, {"player_in": "h6", "player_out": "h2"}])
    three = shot(40, pid="h4", value=3, conf=0.6)
    ok = shot(50, conf=0.95)
    confirmed = shot(60, conf=0.1, status=EventStatus.CONFIRMED)
    flag = ev(EventType.FLAG, 70, kind="x", message="m", ref=None)
    q = correction_queue([starters, foul, reb, sub, three, ok, confirmed, flag], r)
    kinds = [i.event.type for i in q]
    assert kinds == [EventType.FLAG, EventType.SHOT, EventType.SUBSTITUTION, EventType.REBOUND, EventType.FOUL]
    assert ok.id not in {i.event.id for i in q} and confirmed.id not in {i.event.id for i in q}
    it = q[1]
    assert (it.clip_start, it.clip_end) == (37.0, 42.0)
    assert 1 <= len(it.suggestions) <= 3
    assert it.suggestions[0].changes["payload"]["value"] == 2  # 3 <-> 2
    assert any(s.changes["payload"].get("made") is False for s in it.suggestions if "payload" in s.changes)
    sub_item = q[2]
    assert sub_item.suggestions[0].changes["payload"] == {"player_in": "h7", "player_out": "h1"}
    reb_item = q[3]
    assert all(s.changes.get("player_id") != "h3" for s in reb_item.suggestions)
    assert q[1].impact > q[2].impact > q[3].impact > q[4].impact or q[0].impact > 0


def test_queue_order_made_three_beats_sub_beats_rebound_same_confidence():
    evs = [ev(EventType.REBOUND, 1, pid="h1", conf=0.5, offensive=True),
           ev(EventType.SUBSTITUTION, 2, conf=0.5, player_in="h2", player_out="h1"),
           shot(3, value=3, conf=0.5), shot(4, value=2, conf=0.5),
           ev(EventType.FOUL, 5, pid="h1", conf=0.5, kind="personal", fouled_id=None)]
    q = correction_queue(evs, roster())
    assert [i.event.type for i in q] == [EventType.SHOT, EventType.SHOT, EventType.SUBSTITUTION,
                                         EventType.REBOUND, EventType.FOUL]
    assert q[0].event.payload["value"] == 3


def test_apply_review_variants():
    s = shot(10, value=2, conf=0.5)
    q = correction_queue([s], roster())
    c = apply_review(s, "confirm")
    assert c.status == EventStatus.CONFIRMED and c.supersedes == s.id and c.confidence == 1.0
    fixed = apply_review(s, q[0].suggestions[0])
    assert fixed.payload["value"] == 3 and fixed.payload["made"] is True and fixed.supersedes == s.id
    v = apply_review(s, "reject")
    assert v.type == EventType.VOID and v.supersedes == s.id
    assert effective_events([s, c]) == [c]
    assert effective_events([s, v]) == []
    with pytest.raises(ValueError):
        apply_review(s, "nonsense")


def test_reconcile_flags_diff_but_ignores_fta():
    evs = [shot(1, "h1", 3), shot(2, "h1", 2), shot(3, "h1", 2, made=False),
           ev(EventType.FREE_THROW, 4, pid="h1", made=True, attempt=1, of=2),
           ev(EventType.FREE_THROW, 5, pid="h1", made=False, attempt=2, of=2),  # FTA nicht auf dem Bogen
           ev(EventType.FOUL, 6, pid="h2", kind="personal", fouled_id=None)]
    ok = {"players": {"h1": {"points": 6, "fouls": 0, "ftm": 1}, "h2": {"points": 0, "fouls": 1, "ftm": 0}},
          "home": 6, "away": 0}
    assert reconcile_scoresheet(evs, ok) == []
    bad = {"players": {"h1": {"points": 7, "fouls": 0, "ftm": 1}, "h2": {"points": 0, "fouls": 2, "ftm": 0}}}
    flags = reconcile_scoresheet(evs, bad)
    assert {(f.payload["player_id"], f.payload["metric"]) for f in flags} == {("h1", "points"), ("h2", "fouls")}
    assert all(f.type == EventType.FLAG and f.payload["kind"] == "scoresheet_mismatch" for f in flags)


def test_finalize_refuses_then_force():
    store = EventStore()
    store.put_game(Game(id=G, home_team_id="H", away_team_id="A"))
    store.append(shot(1, conf=0.3))
    store.append(shot(2, conf=0.9))
    with pytest.raises(FinalizationError):
        finalize(store, G, "coach")
    assert store.get_game(G).status != GameStatus.FINAL
    fin = finalize(store, G, "coach", force=True)
    assert fin.type == EventType.GAME_FINALIZED and fin.payload["by"] == "coach"
    assert store.get_game(G).status == GameStatus.FINAL

    store2 = EventStore()
    store2.put_game(Game(id=G, home_team_id="H", away_team_id="A"))
    low = store2.append(shot(1, conf=0.3))
    store2.append(apply_review(low, "confirm"))
    finalize(store2, G, "coach")
    assert store2.get_game(G).status == GameStatus.FINAL


def test_console_foul_input():
    store = EventStore()
    f = ConsoleFoulInput(G, store).foul("h1", "home", FoulKind.TECHNICAL, fouled_id="a2", video_ts=12.0)
    assert f.type == EventType.FOUL and f.source == EventSource.CONSOLE
    assert f.status == EventStatus.CONFIRMED and f.payload["fouled_id"] == "a2"
    assert f.payload["kind"] == FoulKind.TECHNICAL and f.seq == 1


def test_accuracy_report():
    auto = {"p1": {"pts": 10, "reb": 4}, "p2": {"pts": 5, "reb": 3}}
    ref = {"p1": {"pts": 10, "reb": 5}, "p2": {"pts": 5, "reb": 3}}
    r = accuracy_report(auto, ref)
    assert r["per_category"]["pts"]["exact_share"] == 1.0
    assert r["per_category"]["reb"]["exact_share"] == 0.5 and r["per_category"]["reb"]["mae"] == 0.5
    assert r["per_player"]["p1"]["exact_share"] == 0.5
    assert r["overall"]["exact_share"] == 0.75 and r["overall"]["mae"] == 0.25


def test_event_f1_tolerance():
    truth = [shot(10.0), shot(20.0), ev(EventType.STEAL, 30.0, pid="h1")]
    pred = [shot(10.8), shot(22.0), ev(EventType.STEAL, 30.2, pid="h1"), ev(EventType.BLOCK, 40.0, pid="a1")]
    r = event_f1(pred, truth, tol_s=1.0)
    assert r["SHOT"]["tp"] == 1 and r["SHOT"]["fp"] == 1 and r["SHOT"]["fn"] == 1
    assert r["SHOT"]["f1"] == pytest.approx(0.5)
    assert r["STEAL"]["f1"] == 1.0 and r["BLOCK"]["precision"] == 0.0
    assert event_f1(pred, truth, tol_s=2.5)["SHOT"]["f1"] == 1.0
    p2 = [ev(EventType.STEAL, 30.0, pid="h9")]
    assert event_f1(p2, truth[2:], match_player=True)["STEAL"]["tp"] == 0


# ---------------------------------------------------------- Highlights
def game_events():
    return [
        # frueher Layup, Q1 deutlich
        shot(60, "h1", 2, period=1, clock=500.0),
        ev(EventType.SCOREBOARD, 61, home=2, away=0, clock=500.0, period=1, running=True),
        # Q4 letzte Minute, knappes Spiel, Dreier
        ev(EventType.SCOREBOARD, 2000, home=60, away=61, clock=50.0, period=4, running=True),
        shot(2010, "h2", 3, period=4, clock=45.0),
    ]


def test_clutch_three_scores_higher_than_early_layup():
    evs = game_events()
    ctx = build_contexts(evs)
    layup, three = evs[0], evs[3]
    assert ctx[three.id].clutch_time and ctx[three.id].close_game and not ctx[layup.id].clutch_time
    assert highlight_score(three, ctx[three.id]) > 2 * highlight_score(layup, ctx[layup.id])
    ranked = score_events(evs)
    assert ranked[0].event.id == three.id


def test_confidence_rarity_and_run():
    evs = [shot(10 + i * 10, "h1", 3, clock=500 - i * 10) for i in range(3)]  # 9:0 Lauf
    c = build_contexts(evs, history={"h1": {"three_made": 0}})
    assert c[evs[2].id].run_points == 9 and c[evs[0].id].prior_count == 0
    base = build_contexts(evs)
    assert highlight_score(evs[0], c[evs[0].id]) > highlight_score(evs[0], base[evs[0].id])
    from courtvision.highlights.scoring import HighlightContext
    assert highlight_score(evs[2], HighlightContext(run_points=9)) > highlight_score(evs[2], HighlightContext(run_points=7))
    low = evs[0].model_copy(update={"confidence": 0.5})
    assert highlight_score(low, c[low.id]) < highlight_score(evs[0], c[evs[0].id])
    assert low.id not in {s.event.id for s in score_events([low, evs[1]], threshold=0.8)}
    conf = low.model_copy(update={"status": EventStatus.CONFIRMED})
    assert conf.id in {s.event.id for s in score_events([conf])}


def test_and_one_and_fastbreak_bonus():
    plain = shot(10, "h1", 2)
    s = shot(100, "h1", 2)
    foul = ev(EventType.FOUL, 100.5, team="away", pid="a1", kind="personal", fouled_id="h1")
    ft = ev(EventType.FREE_THROW, 103, pid="h1", made=True, attempt=1, of=1)
    steal = ev(EventType.STEAL, 200, pid="h3")
    fb = shot(204, "h3", 2)
    c = build_contexts([plain, s, foul, ft, steal, fb])
    assert c[s.id].and_one and c[ft.id].and_one and not c[plain.id].and_one
    assert c[steal.id].fastbreak and c[fb.id].fastbreak
    assert highlight_score(s, c[s.id]) > highlight_score(plain, c[plain.id])
    assert highlight_score(steal, c[steal.id]) > highlight_score(
        ev(EventType.STEAL, 1, pid="h3"), None)


def test_injury_scene_excluded():
    inj = ev(EventType.FLAG, 105, kind="injury", message="player down", ref=None)
    near = shot(100, "h1", 3)
    far = shot(200, "h2", 3)
    assert near.id not in {s.event.id for s in score_events([near, far, inj])}
    assert far.id in {s.event.id for s in score_events([near, far, inj])}
    scenes = build_scenes([near, far, inj])
    assert all(near.id not in sc.event_ids for sc in scenes) and any(far.id in sc.event_ids for sc in scenes)
    ctx = build_contexts([near, inj])
    assert ctx[near.id].injury and highlight_score(near, ctx[near.id]) == 0.0


def big_game(n=14):
    """n Szenen beider Teams ueber das Spiel verteilt."""
    evs = [ev(EventType.GAME_START, 0)]
    for i in range(n):
        t = 100 + i * 90
        team = "home" if i % 2 == 0 else "away"
        pid = f"{team[0]}{i % 5 + 1}"
        evs.append(ev(EventType.POSSESSION, t - 7, team=team, start_ts=t - 7, end_ts=t))
        if i % 4 == 3:
            evs.append(ev(EventType.BLOCK, t, team=team, pid=pid, shot_id="x", shooter_id=None))
        else:
            evs.append(shot(t, pid, 3 if i % 3 == 0 else 2, period=1 + i // 4, clock=600.0 - 10 * i))
            evs[-1] = evs[-1].model_copy(update={"team": team})
    return evs


def test_scene_bounds_and_formats():
    evs = big_game()
    scenes = build_scenes(evs)
    assert len(scenes) == 14
    s0 = scenes[0]
    assert s0.start == pytest.approx(93.0) and s0.end == pytest.approx(102.0)  # Besitzbeginn .. +2 s

    live = live_clip(scenes)
    assert 8.0 <= live.duration <= 15.0 and live.kind == "live"
    assert "KI-generiert" in live.overlays

    rc = recap(scenes)
    assert 8 <= len(rc.segments) <= 12
    assert 60.0 <= rc.duration <= 120.0
    assert {s.team for s in rc.segments} == {"home", "away"}
    assert [s.start for s in rc.segments] == sorted(s.start for s in rc.segments)

    v = vertical_clip(max(scenes, key=lambda s: s.score))
    assert v.aspect == "9:16" and 10.0 <= v.duration <= 30.0 and v.segments[0].camera == "actor"

    sr = season_reel({"g1": scenes, "g2": [s.__class__(**{**s.__dict__, "game_id": "g2"}) for s in scenes]})
    assert 120.0 <= sr.duration <= 180.0 and {s.game_id for s in sr.segments} == {"g1", "g2"}


def test_fastbreak_scene_starts_at_steal_and_player_reel():
    steal = ev(EventType.STEAL, 200, pid="h3")
    fb = shot(204, "h3", 2)
    ast = ev(EventType.ASSIST, 204.1, pid="h4", shot_id=fb.id)
    blk = ev(EventType.BLOCK, 300, pid="h3", shot_id="z", shooter_id=None)
    miss = shot(400, "h3", made=False)
    evs = [steal, fb, ast, blk, miss]
    scenes = build_scenes(evs)
    fbs = next(s for s in scenes if fb.id in s.event_ids)
    assert fbs.start == 200 and fbs.end == pytest.approx(206.1)
    reel = player_reel(evs, "h3")
    assert reel.kind == "player_reel" and len(reel.segments) == 2  # Fastbreak + Block, kein Fehlwurf
    assert all(s.actor_id == "h3" for s in reel.segments)
    assert player_reel(evs, "h9") is None


def test_filter_publishable_consent():
    ok = Player(id="h1", team_id="H", number=1, name="Ok")
    obj = Player(id="h2", team_id="H", number=2, name="Obj", consent=int(Consent.adult_default() | Consent.OBJECTED))
    minor = Player(id="h3", team_id="H", number=3, name="Kid", is_minor=True, consent=0)
    minor_ok = Player(id="h4", team_id="H", number=4, name="Kid2", is_minor=True,
                      consent=int(Consent.STATS_INTERNAL | Consent.HIGHLIGHTS))
    nohl = Player(id="h5", team_id="H", number=5, name="NoHL", consent=int(Consent.STATS_INTERNAL))
    evs = [shot(100 + 100 * i, f"h{i + 1}", 3) for i in range(5)]
    scenes = build_scenes(evs)
    pl = live_clip(scenes, [ok, obj, minor, minor_ok, nohl])
    plans = [recap(scenes, [ok, obj, minor, minor_ok, nohl])]
    out = filter_publishable(plans, [ok, obj, minor, minor_ok, nohl], channel="team")
    actors = {s.actor_id for s in out[0].segments}
    assert actors == {"h1", "h4"}  # Widerspruch, Minderjaehrige ohne Einwilligung, ohne HIGHLIGHTS entfernt
    pub = filter_publishable(plans, [ok, obj, minor, minor_ok, nohl], channel="public")
    assert {s.actor_id for s in pub[0].segments} == {"h1"}  # Minderjaehriger ohne oeffentliche Freigabe
    assert filter_publishable([pl], [obj, minor]) == []
    # Namen im Overlay: h4 hat keine STATS_PUBLIC -> nur Nummer
    seg = next(s for s in out[0].segments if s.actor_id == "h4")
    assert seg.overlay.name is None and seg.overlay.number == 4


def test_overlay_hides_name_without_consent():
    adult = Player(id="a", team_id="H", number=23, name="Max Muster")
    o = build_overlay(adult, (10, 8), 125.0, 2, logo="club.png")
    assert o.name == "Max Muster" and o.clock == "02:05" and o.logo == "club.png"
    assert overlay_text_lines(o)[0] == "KI-generiert" and "#23 Max Muster" in overlay_text_lines(o)
    objected = adult.model_copy(update={"consent": int(Consent.OBJECTED)})
    o2 = build_overlay(objected, (10, 8), 125.0, 2)
    assert o2.name is None and o2.number == 23 and "#23" in overlay_text_lines(o2)
    assert not any("Max" in l for l in overlay_text_lines(o2))
    kid = adult.model_copy(update={"is_minor": True, "consent": 0})
    assert build_overlay(kid).name is None


def test_render_command_builds_argv_without_running():
    scenes = build_scenes(big_game())
    plan = live_clip(scenes, [Player(id="h1", team_id="H", number=1, name="Ok")])
    cfg = CameraConfig()
    n = int(plan.duration * cfg.fps)
    track = virtual_camera([(1900 + i, 1000) for i in range(n)], [[(1800, 1000)]] * n, cfg)
    cmd = render_command(plan, "/in/game.mp4", "/out/clip.mp4", [track])
    assert cmd[0] == "ffmpeg" and cmd[-1] == "/out/clip.mp4" and "-filter_complex" in cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "KI-generiert" in graph and "crop=" in graph and "scale=1920:1080" in graph
    v = vertical_clip(scenes[0])
    vcmd = render_command(v, {"g1": "/in/g.mp4"}, "/out/v.mp4", None, with_audio=False)
    assert "scale=1080:1920" in vcmd[vcmd.index("-filter_complex") + 1]


def test_virtual_camera_inside_frame_and_smooth():
    cfg = CameraConfig()
    n = 200
    # Ball springt abrupt von links nach rechts (Fastbreak) und ausserhalb des gueltigen Bereichs
    ball = [(100.0, 1000.0)] * 60 + [(3800.0, 1100.0)] * 100 + [None] * 40
    players = [[(150.0, 900.0), (300.0, 1200.0)]] * 60 + [[(3700.0, 1000.0), (3600.0, 1300.0)]] * 140
    rects = virtual_camera(ball, players, cfg)
    assert len(rects) == n
    for r in rects:
        assert r.w == 1920 and r.h == 1080
        assert 0 <= r.x <= 3840 - 1920 and 0 <= r.y <= 2160 - 1080
    xs = np.array([r.x for r in rects])
    step = np.abs(np.diff(xs))
    assert step.max() <= cfg.max_pan_speed / cfg.fps + 1e-6  # begrenzte Schwenkgeschwindigkeit
    assert step.max() > 0 and xs[-1] > xs[0] + 1000  # folgt dem Spiel
    acc = np.abs(np.diff(step))
    assert acc.max() < cfg.max_pan_speed / cfg.fps / 2  # sanft: keine Ruckler
    # Vertikal: folgt dem Akteur
    actor = [(3000.0, 1000.0)] * n
    vr = virtual_camera(ball, players, cfg, mode="actor", actor=actor)
    assert vr[0].h < vr[0].w * 2 and vr[0].w / vr[0].h == pytest.approx(9 / 16, abs=0.01)
    assert abs((vr[-1].x + vr[-1].w / 2) - 3000.0) < 5 and 0 <= vr[-1].y <= 2160 - vr[-1].h


# ------------------------------------------------------------- Privacy
def test_consent_manager_rules_and_audit():
    adult = Player(id="a", team_id="H", number=1, name="A")
    kid = Player(id="k", team_id="H", number=2, name="K", is_minor=True)
    teen = Player(id="t", team_id="H", number=3, name="T", is_minor=True)
    t = iter(range(1000))
    cm = ConsentManager([adult, kid, teen], child_age=14, clock=lambda: float(next(t)))
    cm.set_age("k", 10)
    cm.set_age("t", 15)
    assert adult.has(Consent.HIGHLIGHTS)
    assert not kid.has(Consent.STATS_INTERNAL) and team_only_stats(kid)  # ohne Eltern: OBJECTED
    cm.grant_parental("k", Consent.STATS_INTERNAL | Consent.HIGHLIGHTS)
    assert kid.has(Consent.HIGHLIGHTS) and not team_only_stats(kid)  # unter 14: Eltern genuegen
    cm.grant_parental("t", Consent.STATS_INTERNAL | Consent.HIGHLIGHTS | Consent.STATS_PUBLIC)
    assert not teen.has(Consent.HIGHLIGHTS)  # 15 J.: zusaetzlich eigene Einwilligung
    cm.grant_child("t")
    assert teen.has(Consent.HIGHLIGHTS) and can_show_stats_public(teen)
    assert not public_ratings_allowed(teen) and public_ratings_allowed(adult)
    cm.object_("a")
    assert team_only_stats(adult) and not can_show_stats_public(adult)
    cm.withdraw_objection("a")
    cm.revoke("a", Consent.LIVESTREAM)
    assert not adult.has(Consent.LIVESTREAM) and adult.has(Consent.STATS_PUBLIC)
    trail = cm.audit_trail("a")
    assert [r.action for r in trail] == ["object", "withdraw_objection", "revoke"]
    assert trail[0].after == int(Consent.OBJECTED) and trail[0].ts < trail[1].ts
    assert len(cm.audit_trail()) > len(trail)


def test_livestream_allowed():
    game = Game(id=G, home_team_id="H", away_team_id="A")
    ps = [Player(id="1", team_id="H", number=1, name="a"), Player(id="2", team_id="A", number=2, name="b"),
          Player(id="9", team_id="X", number=9, name="other", consent=0)]
    assert livestream_allowed(game, ps)
    ps[1] = ps[1].model_copy(update={"is_minor": True, "consent": int(Consent.STATS_INTERNAL)})
    assert not livestream_allowed(game, ps)


def test_retention_and_biometrics():
    now = datetime(2026, 11, 1)
    yes = [Player(id=str(i), team_id="H", number=i, name="x",
                  consent=int(Consent.adult_default() | Consent.TRAINING_DATA)) for i in range(3)]
    mixed = yes[:2] + [Player(id="9", team_id="H", number=9, name="n")]
    recs = [
        Recording("old", "g1", now - timedelta(days=91), mixed),
        Recording("fresh", "g2", now - timedelta(days=89), mixed),
        Recording("train_old", "g3", now - timedelta(days=200), yes),
        Recording("train_very_old", "g4", now - timedelta(days=800), yes),
        Recording("empty", "g5", now - timedelta(days=100), []),
    ]
    assert {r.id for r in due_for_deletion(recs, now)} == {"old", "train_very_old", "empty"}
    gallery = {"f": [1, 2]}
    register_biometric_store(gallery.clear)
    assert purge_transient_biometrics() >= 1 and gallery == {}


def test_dpia_checklist():
    cl = default_checklist()
    assert len(cl.items) == 6 and not cl.is_complete() and len(cl.pilot_blockers()) == 6
    cl.set_status("dpia", ItemStatus.DONE, owner="dsb")
    assert cl.progress() == pytest.approx(1 / 6) and "dpia" not in {i.id for i in cl.open_items()}
    for k in list(cl.items):
        cl.set_status(k, ItemStatus.DONE)
    assert cl.is_complete() and cl.pilot_blockers() == []


# -------------------------------------------------------------- Business
def test_pricing_and_unit_economics():
    assert invoice(15).total == 199.0
    assert invoice(20).total == 199.0
    inv = invoice(23)
    assert inv.total == 229.0 and inv.lines[1][1] == 30.0
    assert PricingPlan(vat_rate=0.19) and invoice(10, PricingPlan(vat_rate=0.19)).total == 236.81
    assert livestream_split(100.0) == (70.0, 30.0) and livestream_split(100.0, club_share=0.5) == (50.0, 50.0)
    ue = unit_economics()
    assert ue.revenue == 199.0 and ue.operating_cost == 75.0
    assert ue.hardware_per_month == pytest.approx(83.33) and ue.contribution_margin == pytest.approx(40.67)
    assert unit_economics(hardware_per_month=85.0).contribution_margin == 39.0  # Plan: "etwa 39 EUR"
    assert ue.contract_total == 199 * 24
    assert unit_economics(games_per_month=25).revenue == 199 + 5 * 10
    assert unit_economics(premium_subscribers=10).revenue == pytest.approx(199 + 39.9)
    assert 4.0 <= CostPerGame().total_eur <= 6.0
    assert CostPerGame().stream_usd == pytest.approx(3.6)
