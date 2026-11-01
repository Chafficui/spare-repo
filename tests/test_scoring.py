"""Tests fuer EventEngine und SubstitutionDetector (synthetische Observationen)."""
from __future__ import annotations

import pytest

from courtvision.domain.events import Event, EventSource, EventStatus, EventType as T
from courtvision.domain.models import Player
from courtvision.domain.observations import (
    BallTouch, ClockState, Formation, FrameState, OutOfBounds, PlayerState,
    Possession, RimMake, ShotRelease,
)
from courtvision.scoring import EngineConfig, EventEngine, SubstitutionDetector


# ---------------------------------------------------------------- builders
def P(ts, track, team, pid=None, conf=1.0):
    return Possession(ts, track, team, pid, conf)


def FREE(ts):
    return Possession(ts, None, None)


def SHOT(ts, track, team, pid=None, x=8.0, y=0.0, three=0.0, basket="right", conf=1.0):
    return ShotRelease(ts, track, team, x, y, basket, pid, three, conf)


def MAKE(ts, basket="right"):
    return RimMake(ts, basket)


def TOUCH(ts, track, team, pid=None, abrupt=False):
    return BallTouch(ts, track, team, pid, abrupt)


def OOB(ts, track=None, team=None, pid=None):
    return OutOfBounds(ts, track, team, pid)


def run(obs, rosters=None, config=None, flush=True):
    eng = EventEngine("g1", rosters, config)
    evs: list[Event] = []
    for o in obs:
        evs += eng.feed(o)
    if flush:
        evs += eng.flush()
    return eng, evs


def of(evs, typ):
    return [e for e in evs if e.type == typ]


def types(evs):
    return [e.type for e in evs]


# ------------------------------------------------------------ field goals
def test_made_two_payload_clip_and_defaults():
    eng, evs = run([ClockState(0, 1, 540.0, True), P(0.0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1", x=8.0, y=1.0, three=0.0),
                    MAKE(2.0)])
    (shot,) = of(evs, T.SHOT)
    assert shot.payload == {"value": 2, "made": True, "x": 8.0, "y": 1.0, "three_prob": 0.0}
    assert shot.player_id == "h1" and shot.team == "home" and shot.track_id == 1
    assert shot.video_ts == 1.0 and shot.period == 1 and shot.clock == 540.0
    assert shot.source == EventSource.OPTICAL and shot.status == EventStatus.PROVISIONAL
    assert shot.clip_start == 0.0 and shot.clip_end == 4.0  # Besitzbeginn .. Make + 2 s
    assert 0.0 < shot.confidence <= 1.0
    assert eng.optical_score == {"home": 2, "away": 0}
    (pos,) = of(evs, T.POSSESSION)
    assert pos.payload == {"start_ts": 0.0, "end_ts": 1.0} and pos.player_id == "h1"


def test_made_three_and_ambiguous_confidence():
    eng, evs = run([P(0, 1, "away", "a1"), SHOT(1.0, 1, "away", "a1", x=-9.0, three=0.95, basket="left"),
                    MAKE(1.8, "left")])
    (shot,) = of(evs, T.SHOT)
    assert shot.payload["value"] == 3 and shot.payload["made"] is True
    assert eng.optical_score == {"home": 0, "away": 3}
    _, evs2 = run([P(0, 1, "away", "a1"), SHOT(1.0, 1, "away", "a1", three=0.55), MAKE(1.8)])
    assert of(evs2, T.SHOT)[0].payload["value"] == 3
    assert of(evs2, T.SHOT)[0].confidence < shot.confidence


def test_rim_make_other_basket_or_too_late_is_miss():
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), MAKE(1.5, "left")])
    assert of(evs, T.SHOT)[0].payload["made"] is False
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), MAKE(6.0)])
    assert of(evs, T.SHOT)[0].payload["made"] is False


def test_unknown_player_keeps_track_and_lowers_confidence():
    _, a = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), MAKE(2.0)])
    _, b = run([P(0, 1, "home"), SHOT(1.0, 1, "home"), MAKE(2.0)])
    sb = of(b, T.SHOT)[0]
    assert sb.player_id is None and sb.track_id == 1
    assert sb.confidence < of(a, T.SHOT)[0].confidence


def test_track_identity_learned_from_frames():
    fr = FrameState(0.0, (PlayerState(1, "home", 8, 0, "h1", 0.9),))
    _, evs = run([fr, P(0.1, 1, "home"), SHOT(1.0, 1, "home"), MAKE(2.0)])
    assert of(evs, T.SHOT)[0].player_id == "h1"


# --------------------------------------------------------------- rebounds
def test_miss_defensive_rebound():
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), FREE(1.1), P(3.5, 5, "away", "a5")])
    (shot,) = of(evs, T.SHOT)
    assert shot.payload["made"] is False
    (rb,) = of(evs, T.REBOUND)
    assert rb.payload == {"offensive": False} and rb.player_id == "a5" and rb.team == "away"
    assert rb.clip_start == 0.0 and rb.clip_end == 5.5


def test_miss_offensive_rebound():
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), P(3.0, 2, "home", "h2")])
    (rb,) = of(evs, T.REBOUND)
    assert rb.payload == {"offensive": True} and rb.player_id == "h2" and rb.team == "home"
    assert not of(evs, T.PASS) and not of(evs, T.ASSIST)  # Rebound ist kein Pass vom Werfer


def test_miss_then_out_of_bounds_is_team_rebound_for_next_possession():
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), OOB(2.5, 2, "home", "h2"),
                  P(8.0, 5, "away", "a5")])
    (rb,) = of(evs, T.REBOUND)
    assert rb.player_id is None and rb.team == "away" and rb.payload == {"offensive": False}
    assert not of(evs, T.TURNOVER)


def test_team_rebound_offensive_when_shooting_team_gets_ball():
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), OOB(2.5, 5, "away", "a5"),
                  P(7.0, 3, "home", "h3")])
    (rb,) = of(evs, T.REBOUND)
    assert rb.team == "home" and rb.player_id is None and rb.payload["offensive"] is True


def test_team_rebound_fallback_on_flush():
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), OOB(2.5, 1, "home", "h1")])
    (rb,) = of(evs, T.REBOUND)
    assert rb.team == "away" and rb.player_id is None and rb.confidence < 0.5


def test_scramble_lowers_rebound_confidence():
    def scenario(crowd):
        players = tuple(PlayerState(i, "away" if i % 2 else "home", 11.5 + 0.1 * (i - 10), 0.1 * (i - 10)) for i in range(10, 10 + crowd))
        fr = FrameState(3.4, players, ball_x=11.5, ball_y=0.0)
        return of(run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), fr, P(3.5, 11, "away", "a1")])[1], T.REBOUND)[0]
    assert scenario(5).confidence < scenario(1).confidence


# ------------------------------------------------------- assist / passes
def test_assist_for_quick_score_after_pass():
    _, evs = run([P(0, 1, "home", "h1"), FREE(1.0), P(1.5, 2, "home", "h2"),
                  SHOT(2.5, 2, "home", "h2"), MAKE(3.5)])
    (shot,) = of(evs, T.SHOT)
    (ast,) = of(evs, T.ASSIST)
    assert ast.player_id == "h1" and ast.team == "home"
    assert ast.payload == {"shot_id": shot.id}
    (ps,) = of(evs, T.PASS)
    assert ps.player_id == "h1" and ps.payload == {"to_id": "h2"}
    assert shot.clip_start == 0.0


def test_no_assist_when_receiver_holds_long():
    _, evs = run([P(0, 1, "home", "h1"), FREE(1.0), P(1.5, 2, "home", "h2"),
                  SHOT(6.0, 2, "home", "h2"), MAKE(7.0)])
    assert of(evs, T.PASS) and not of(evs, T.ASSIST)
    assert of(evs, T.SHOT)[0].payload["made"] is True


def test_no_assist_on_miss_and_window_configurable():
    _, evs = run([P(0, 1, "home", "h1"), FREE(1.0), P(1.5, 2, "home", "h2"),
                  SHOT(2.5, 2, "home", "h2"), FREE(3.0), P(6.0, 5, "away", "a5")])
    assert not of(evs, T.ASSIST)
    obs = [P(0, 1, "home", "h1"), FREE(1.0), P(1.5, 2, "home", "h2"), SHOT(5.0, 2, "home", "h2"), MAKE(5.8)]
    assert not of(run(obs)[1], T.ASSIST)
    assert of(run(obs, config=EngineConfig(assist_window_s=4.0))[1], T.ASSIST)


# ----------------------------------------------------------------- steals
def test_steal_with_touch():
    _, evs = run([P(0, 1, "home", "h1"), FREE(1.0), TOUCH(1.1, 7, "away", "a7"), P(1.3, 7, "away", "a7")])
    (st,) = of(evs, T.STEAL)
    assert st.player_id == "a7" and st.team == "away" and st.payload == {"victim_id": "h1"}
    (to,) = of(evs, T.TURNOVER)
    assert to.player_id == "h1" and to.team == "home" and to.payload == {"kind": "steal"}


def test_turnover_bad_pass_without_touch_has_lower_confidence():
    _, evs = run([P(0, 1, "home", "h1"), FREE(1.0), P(2.5, 7, "away", "a7")])
    assert not of(evs, T.STEAL)
    (to,) = of(evs, T.TURNOVER)
    assert to.payload == {"kind": "bad_pass"} and to.player_id == "h1" and to.confidence <= 0.5
    _, evs2 = run([P(0, 1, "home", "h1"), FREE(1.0), TOUCH(1.1, 7, "away", "a7"), P(1.3, 7, "away", "a7")])
    assert of(evs2, T.STEAL)[0].confidence > to.confidence


def test_direct_ripoff_is_low_confidence_steal():
    _, evs = run([P(0, 1, "home", "h1"), P(1.0, 7, "away", "a7")])
    (st,) = of(evs, T.STEAL)
    assert st.player_id == "a7" and st.confidence < 0.7


def test_steal_touch_by_teammate_of_victim_is_ignored():
    _, evs = run([P(0, 1, "home", "h1"), FREE(1.0), TOUCH(1.1, 2, "home", "h2"), P(2.5, 7, "away", "a7")])
    assert not of(evs, T.STEAL) and of(evs, T.TURNOVER)[0].payload["kind"] == "bad_pass"


def test_fastbreak_clip_starts_at_steal():
    _, evs = run([P(0, 1, "home", "h1"), FREE(5.0), P(8.0, 2, "home", "h2"), FREE(9.0),
                  TOUCH(10.0, 7, "away", "a7"), P(10.2, 7, "away", "a7"),
                  SHOT(13.0, 7, "away", "a7", x=-9.0, basket="left"), MAKE(13.8, "left")])
    (st,) = of(evs, T.STEAL)
    (shot,) = of(evs, T.SHOT)
    assert shot.clip_start == 10.0 and shot.clip_end == pytest.approx(15.8)
    assert st.clip_start == 0.0  # Szenenbeginn des Opfer-Teams (Pass h1->h2 setzt ihn nicht zurueck)
    assert st.clip_end == pytest.approx(12.2)


# ----------------------------------------------------------- out of bounds
def test_out_of_bounds_turnover_for_last_touch():
    _, evs = run([P(0, 1, "home", "h1"), OOB(1.5, 1, "home", "h1")])
    (to,) = of(evs, T.TURNOVER)
    assert to.payload == {"kind": "out_of_bounds"} and to.player_id == "h1" and to.team == "home"
    assert to.clip_start == 0.0 and to.clip_end == 3.5


def test_out_of_bounds_after_pass_in_flight_and_deflection():
    _, evs = run([P(0, 1, "home", "h1"), FREE(0.8), OOB(1.5, 1, "home", "h1")])
    assert of(evs, T.TURNOVER)[0].player_id == "h1"
    # Ball von Verteidiger ins Aus gelenkt -> kein Turnover des Angriffs
    _, evs = run([P(0, 1, "home", "h1"), FREE(0.8), OOB(1.5, 7, "away", "a7")])
    assert not of(evs, T.TURNOVER)


def test_inbound_after_out_of_bounds_is_not_a_turnover():
    _, evs = run([P(0, 1, "home", "h1"), OOB(1.5, 1, "home", "h1"), P(8.0, 7, "away", "a7")])
    assert len(of(evs, T.TURNOVER)) == 1 and not of(evs, T.STEAL)


# ------------------------------------------------------------------ block
def test_block_payload_and_missed_shot():
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), TOUCH(1.3, 7, "away", "a7", abrupt=True),
                  P(2.5, 5, "away", "a5")])
    (shot,) = of(evs, T.SHOT)
    (blk,) = of(evs, T.BLOCK)
    assert shot.payload["made"] is False
    assert blk.player_id == "a7" and blk.team == "away"
    assert blk.payload == {"shot_id": shot.id, "shooter_id": "h1"}
    (rb,) = of(evs, T.REBOUND)
    assert rb.player_id == "a5" and rb.payload["offensive"] is False


def test_touch_by_own_team_or_non_abrupt_or_late_is_not_a_block():
    for t in (TOUCH(1.3, 2, "home", "h2", abrupt=True), TOUCH(1.3, 7, "away", "a7", abrupt=False),
              TOUCH(2.5, 7, "away", "a7", abrupt=True)):
        _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), t, P(4.0, 5, "away", "a5")])
        assert not of(evs, T.BLOCK)


# ------------------------------------------------------------ free throws
def FT(ts, track=1, pid="h1"):
    return Formation(ts, "free_throw", track, "home", pid, "right")


def test_free_throw_series_last_missed_with_rebound():
    _, evs = run([ClockState(5, 2, 100.0, False), FT(10.0),
                  SHOT(12.0, 1, "home", "h1", x=4.6), MAKE(13.0),
                  P(14.0, 1, "home", "h1"),  # Ball vom Schiri -> kein Rebound
                  SHOT(18.0, 1, "home", "h1", x=4.6),
                  P(22.0, 5, "away", "a5")])
    fts = of(evs, T.FREE_THROW)
    assert [e.payload for e in fts] == [{"made": True, "attempt": 1, "of": 2}, {"made": False, "attempt": 2, "of": 2}]
    assert all(e.player_id == "h1" and e.team == "home" for e in fts)
    assert not of(evs, T.SHOT)
    (rb,) = of(evs, T.REBOUND)
    assert rb.player_id == "a5" and rb.payload == {"offensive": False}


def test_free_throw_first_miss_has_no_rebound_when_second_follows():
    eng, evs = run([FT(10.0), SHOT(12.0, 1, "home", "h1"), P(14.0, 1, "home", "h1"),
                    SHOT(16.0, 1, "home", "h1"), MAKE(17.0), P(19.0, 5, "away", "a5")])
    fts = of(evs, T.FREE_THROW)
    assert [(e.payload["made"], e.payload["attempt"]) for e in fts] == [(False, 1), (True, 2)]
    assert not of(evs, T.REBOUND)
    assert eng.optical_score == {"home": 1, "away": 0}


def test_free_throw_offensive_rebound_and_three_shot_series_of_config():
    cfg = EngineConfig(ft_default_of=3)
    _, evs = run([FT(10.0), SHOT(12.0, 1, "home", "h1"), SHOT(16.0, 1, "home", "h1"),
                  P(19.0, 2, "home", "h2")], config=cfg)
    assert [e.payload["of"] for e in of(evs, T.FREE_THROW)] == [3, 3]
    assert of(evs, T.REBOUND)[0].payload == {"offensive": True}


def test_field_goal_after_free_throw_series_is_a_shot():
    _, evs = run([FT(10.0), SHOT(12.0, 1, "home", "h1"), MAKE(13.0), ClockState(20, 2, 90.0, True),
                  P(21.0, 2, "home", "h2"), SHOT(22.0, 2, "home", "h2"), MAKE(23.0)])
    assert len(of(evs, T.FREE_THROW)) == 1 and len(of(evs, T.SHOT)) == 1


# -------------------------------------------------------------- jump ball
def test_jump_ball_is_not_a_turnover():
    _, evs = run([P(-5.0, 7, "away", "a7"), FREE(-4.5), Formation(0.0, "jump_ball"), P(1.0, 1, "home", "h1"),
                  FREE(2.0), P(3.0, 7, "away", "a7")])
    (jb,) = of(evs, T.JUMP_BALL)
    assert jb.payload == {"winner_team": "home"} and jb.video_ts == 0.0
    assert [e.payload["kind"] for e in of(evs, T.TURNOVER)] == ["bad_pass"]  # nur der spaetere echte Ballverlust
    assert not [e for e in of(evs, T.TURNOVER) if e.video_ts < 2.0]


def test_jump_ball_without_possession_flushes_unknown_winner():
    _, evs = run([Formation(0.0, "jump_ball")])
    assert of(evs, T.JUMP_BALL)[0].payload == {"winner_team": None}


def test_clock_stop_breaks_possession_chain():
    _, evs = run([ClockState(0, 1, 300.0, True), P(0.5, 1, "home", "h1"), ClockState(1.0, 1, 299.0, False),
                  P(5.0, 7, "away", "a7")])
    assert not of(evs, T.TURNOVER) and not of(evs, T.STEAL)


# ------------------------------------------------------------ substitutions
ROSTER = {
    "home": [Player(id=f"h{i}", team_id="H", number=i, name=f"H{i}") for i in range(1, 9)],
    "away": [Player(id=f"a{i}", team_id="A", number=i, name=f"A{i}") for i in range(1, 9)],
}
HOME_POS = {1: (-6, -3), 2: (-3, 4), 3: (0, 0), 4: (3, -4), 5: (6, 2)}
AWAY_POS = {11: (-5, 3), 12: (-2, -4), 13: (1, 3), 14: (4, -2), 15: (7, 0)}


def frame(ts, home=None, away=None, ids=True):
    ps = []
    for t, (x, y, *rest) in (home if home is not None else {k: v for k, v in HOME_POS.items()}).items():
        pid = rest[0] if rest else (f"h{t}" if ids and t <= 5 else None)
        ps.append(PlayerState(t, "home", x, y, pid, 0.9 if pid else 0.0))
    for t, (x, y) in (away if away is not None else AWAY_POS).items():
        ps.append(PlayerState(t, "away", x, y, f"a{t - 10}" if ids else None, 0.9 if ids else 0.0))
    return FrameState(ts, tuple(ps))


def home_after_sub(ts, pid6=None):
    d = {k: v for k, v in HOME_POS.items() if k != 3}
    d[6] = (0.0, 6.0, pid6) if pid6 else (0.0, 6.0)
    return d


def sub_obs(pid6=None, running_during=False):
    """Uhr laeuft, Stopp bei 10, Track 3 geht in Wechselzone, Track 6 kommt, Neustart bei 16."""
    obs = [ClockState(0.0, 1, 500.0, True)]
    obs += [frame(float(t)) for t in range(0, 10)]
    if not running_during:
        obs.append(ClockState(10.0, 1, 480.0, False))
    obs += [frame(11.0, {**HOME_POS, 6: (0.0, 8.8)}), frame(11.5, {**HOME_POS, 3: (0.0, 8.0), 6: (0.0, 8.8)})]
    obs += [frame(12.0, {**{k: v for k, v in HOME_POS.items() if k != 3}, 3: (0.0, 8.3), 6: (0.0, 6.0)}),
            frame(14.0, {**{k: v for k, v in HOME_POS.items() if k != 3}, 3: (0.5, 8.3), 6: (0.0, 6.0)})]
    obs.append(frame(15.0, {**{k: v for k, v in HOME_POS.items() if k != 3}, 3: (0.5, 8.3), 6: (0.0, 6.0, pid6) if pid6 else (0.0, 6.0)}))
    if not running_during:
        obs.append(ClockState(16.0, 1, 480.0, True))
    return obs


def start_engine():
    eng = EventEngine("g1", ROSTER)
    evs = eng.set_starters("home", ["h1", "h2", "h3", "h4", "h5"])
    evs += eng.set_starters("away", ["a1", "a2", "a3", "a4", "a5"])
    return eng, evs


def feed_all(eng, obs):
    out = []
    for o in obs:
        out += eng.feed(o)
    return out


def running_frames(t0, t1, home, step=1.0):
    t = t0
    while t <= t1:
        yield frame(t, home)
        t += step


def test_starters_event_and_on_court():
    eng, evs = start_engine()
    assert [e.type for e in evs] == [T.STARTERS, T.STARTERS]
    assert evs[0].team == "home" and evs[0].payload == {"players": ["h1", "h2", "h3", "h4", "h5"]}
    assert eng.on_court("away") == {"a1", "a2", "a3", "a4", "a5"}
    with pytest.raises(ValueError):
        eng.subs.set_starters("home", ["h1", "h2"])


def test_substitution_during_stopped_clock_resolved_by_number():
    eng, _ = start_engine()
    evs = feed_all(eng, sub_obs(pid6="h7"))
    (sub,) = of(evs, T.SUBSTITUTION)
    assert sub.payload == {"player_in": "h7", "player_out": "h3"}
    assert sub.team == "home" and sub.player_id == "h7" and sub.track_id == 6
    assert sub.confidence >= 0.8 and sub.video_ts == pytest.approx(12.0)
    assert eng.on_court("home") == {"h1", "h2", "h4", "h5", "h7"}


def test_court_change_while_clock_running_is_no_substitution():
    eng, _ = start_engine()
    evs = feed_all(eng, sub_obs(pid6="h7", running_during=True))
    evs += feed_all(eng, list(running_frames(16.0, 60.0, home_after_sub(0, "h7"))))
    evs += eng.flush()
    assert not of(evs, T.SUBSTITUTION)
    assert eng.on_court("home") == {"h1", "h2", "h3", "h4", "h5"}


def test_substitution_identity_resolved_later():
    eng, _ = start_engine()
    evs = feed_all(eng, sub_obs())
    assert not of(evs, T.SUBSTITUTION)  # Zugang bekannt, Identitaet offen
    evs = feed_all(eng, list(running_frames(16.5, 24.5, home_after_sub(0))))
    assert not of(evs, T.SUBSTITUTION)
    evs = feed_all(eng, [frame(25.0, home_after_sub(0, "h8"))])
    (sub,) = of(evs, T.SUBSTITUTION)
    assert sub.payload == {"player_in": "h8", "player_out": "h3"} and sub.confidence >= 0.8
    assert eng.on_court("home") == {"h1", "h2", "h4", "h5", "h8"}
    assert not of(feed_all(eng, list(running_frames(26.0, 80.0, home_after_sub(0, "h8")))), T.SUBSTITUTION)


def test_unresolved_substitution_fallback_after_30s_game_time():
    eng, _ = start_engine()
    evs = feed_all(eng, sub_obs())
    evs += feed_all(eng, list(running_frames(17.0, 45.0, home_after_sub(0))))  # 29 s Spielzeit
    assert not of(evs, T.SUBSTITUTION)
    evs = feed_all(eng, list(running_frames(46.0, 48.0, home_after_sub(0))))
    (sub,) = of(evs, T.SUBSTITUTION)
    assert sub.payload["player_in"] == "unknown" and sub.payload["player_out"] == "h3"
    assert sub.payload["candidates"] == ["h6", "h7", "h8"]
    assert sub.confidence <= 0.3 and sub.track_id == 6 and sub.player_id is None


def test_stopped_clock_time_does_not_count_for_fallback():
    eng, _ = start_engine()
    evs = feed_all(eng, sub_obs())
    evs += feed_all(eng, [ClockState(20.0, 1, 470.0, False), frame(100.0, home_after_sub(0))])
    assert not of(evs, T.SUBSTITUTION)


def test_fallback_candidates_exclude_disqualified_and_late_id_supersedes():
    eng, _ = start_engine()
    eng.disqualify("h6")
    evs = feed_all(eng, sub_obs())
    evs += feed_all(eng, list(running_frames(17.0, 48.0, home_after_sub(0))))
    (sub,) = of(evs, T.SUBSTITUTION)
    assert sub.payload["candidates"] == ["h7", "h8"]
    late = feed_all(eng, [frame(49.0, home_after_sub(0, "h7"))])
    (fix,) = of(late, T.SUBSTITUTION)
    assert fix.supersedes == sub.id and fix.payload["player_in"] == "h7" and fix.payload["player_out"] == "h3"


def test_disqualified_player_cannot_enter():
    eng, _ = start_engine()
    eng.disqualify("h7")
    evs = feed_all(eng, sub_obs(pid6="h7"))
    assert not of(evs, T.SUBSTITUTION) and of(evs, T.FLAG)


def test_unresolved_substitution_flushed_at_end_of_stream():
    eng, _ = start_engine()
    feed_all(eng, sub_obs())
    evs = eng.flush()
    (sub,) = of(evs, T.SUBSTITUTION)
    assert sub.payload["player_in"] == "unknown"


def test_substitution_detector_standalone_and_free_throw_stoppage():
    det = SubstitutionDetector(ROSTER)
    det.set_starters("home", ["h1", "h2", "h3", "h4", "h5"])
    evs = []
    for t in range(0, 6):
        evs += det.feed(frame(float(t)))  # Uhr unbekannt: kein Stillstand
    evs += det.set_stoppage("free_throw", True)
    evs += det.feed(frame(7.0, {**HOME_POS, 6: (0.0, 8.8)}))
    evs += det.feed(frame(8.0, {**{k: v for k, v in HOME_POS.items() if k != 3}, 3: (0.0, 8.0), 6: (0.0, 6.0, "h6")}))
    evs += det.feed(frame(9.5, {**{k: v for k, v in HOME_POS.items() if k != 3}, 3: (0.0, 8.0), 6: (0.0, 6.0, "h6")}))
    evs += det.set_stoppage("free_throw", False)
    (sub,) = of(evs, T.SUBSTITUTION)
    assert sub.payload == {"player_in": "h6", "player_out": "h3"}


# ------------------------------------------------------------- general
def test_all_events_validate_and_share_conventions():
    obs = [ClockState(0, 1, 500.0, True), P(0, 1, "home", "h1"), FREE(1.0), P(1.5, 2, "home", "h2"),
           SHOT(2.5, 2, "home", "h2", three=0.9), MAKE(3.5), P(8.0, 7, "away", "a7"), OOB(9.0, 7, "away", "a7")]
    _, evs = run(obs)
    assert evs
    for e in evs:
        Event.model_validate(e.model_dump())
        assert e.game_id == "g1" and e.source == EventSource.OPTICAL and e.status == EventStatus.PROVISIONAL
        assert 0.0 < e.confidence <= 1.0
        if e.clip_start is not None:
            assert e.clip_start <= e.clip_end


def test_low_confidence_observation_ignored():
    _, evs = run([P(0, 1, "home", "h1"), SHOT(1.0, 1, "home", "h1"), RimMake(2.0, "right", 0.1)])
    assert of(evs, T.SHOT)[0].payload["made"] is False
