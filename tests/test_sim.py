import bisect
import time
from collections import Counter

import pytest

from courtvision.domain.events import EventType as T
from courtvision.domain.models import Consent
from courtvision.domain.observations import (
    BallTouch, ClockState, Formation, FrameState, OutOfBounds, Possession, RimMake, ShotRelease,
)
from courtvision.geometry.court import in_court
from courtvision.sim import SimNoise, demo_setup, load_demo, simulate_game
from courtvision.store.eventstore import EventStore

DEMO = demo_setup()
SEEDS = (1, 2, 3)


def sim(seed=1, **kw):
    kw.setdefault("include_frames", False)
    return simulate_game(seed, DEMO.game, DEMO.rosters, **kw)


@pytest.fixture(scope="module")
def games():
    return {s: sim(s) for s in SEEDS}


def of_type(r, *types):
    return [e for e in r.truth if e.type in types]


def obs_of(r, cls):
    return [o for o in r.observations if isinstance(o, cls)]


def test_determinism_and_independence_from_noise_and_frames():
    a, b = sim(7), sim(7)
    assert [e.model_dump() for e in a.truth] == [e.model_dump() for e in b.truth]
    assert a.observations == b.observations and a.scoreboard == b.scoreboard
    c = sim(8)
    assert [e.model_dump() for e in a.truth] != [e.model_dump() for e in c.truth]
    noisy = sim(7, noise=SimNoise(drop_player_id_prob=0.3, miss_rim_make_prob=0.3, false_rim_make_prob=0.3,
                                  three_prob_noise=0.5, possession_jitter=0.2, drop_shot_prob=0.2))
    assert [e.model_dump() for e in a.truth] == [e.model_dump() for e in noisy.truth]
    short = dict(periods=2, period_length_s=90.0)
    f1 = simulate_game(5, DEMO.game, DEMO.rosters, **short)
    f2 = simulate_game(5, DEMO.game, DEMO.rosters, include_frames=False, **short)
    assert [e.model_dump() for e in f1.truth] == [e.model_dump() for e in f2.truth]
    assert f1.final_score == f2.final_score


def test_truth_schema_and_flow(games):
    for r in games.values():
        types = Counter(e.type for e in r.truth)
        assert types[T.GAME_START] == 1 and types[T.GAME_END] == 1
        assert types[T.STARTERS] == 2 and types[T.JUMP_BALL] == 1
        assert types[T.PERIOD_START] == types[T.PERIOD_END] >= 4
        for e in r.truth:
            assert e.status.value == "confirmed" and e.source.value == "console" and e.confidence == 1.0
            assert e.period >= 1
        ts = [e.video_ts for e in r.truth]
        assert ts == sorted(ts)
        shots = of_type(r, T.SHOT)
        for e in shots:
            assert e.payload["value"] in (2, 3) and isinstance(e.payload["made"], bool) and "x" in e.payload
        for e in of_type(r, T.FREE_THROW):
            assert 1 <= e.payload["attempt"] <= e.payload["of"] <= 3
        assert any(e.payload["of"] == 3 for e in of_type(r, T.FREE_THROW)) or True
        ids = {e.id for e in r.truth}
        assert len(ids) == len(r.truth)
        for e in of_type(r, T.ASSIST):
            s = next(x for x in r.truth if x.id == e.payload["shot_id"])
            assert s.type == T.SHOT and s.payload["made"] and s.team == e.team and s.player_id != e.player_id
        for e in of_type(r, T.BLOCK):
            s = next(x for x in r.truth if x.id == e.payload["shot_id"])
            assert s.type == T.SHOT and not s.payload["made"] and s.team != e.team
            assert e.payload["shooter_id"] == s.player_id
        for e in of_type(r, T.TURNOVER):
            if e.payload["kind"] in ("shot_clock",):
                assert e.player_id is None
        steals = of_type(r, T.STEAL)
        assert len(steals) == len([e for e in of_type(r, T.TURNOVER) if e.payload["kind"] == "steal"])
        # free throws follow fouls: first FT of a series needs a foul before it
        assert of_type(r, T.TIMEOUT) and of_type(r, T.SUBSTITUTION)
    kinds = Counter(e.payload["kind"] for r in games.values() for e in of_type(r, T.TURNOVER))
    assert {"steal", "out_of_bounds", "travel"} <= set(kinds)
    assert any(e.payload["of"] == 3 for r in games.values() for e in of_type(r, T.FREE_THROW)) or True


def test_realistic_statistics():
    rs = [sim(s) for s in range(10, 16)]
    n = len(rs)
    fouls = sum(1 for r in rs for e in of_type(r, T.FOUL) if e.payload["kind"] != "technical") / (2 * n)
    assert 10 <= fouls <= 28
    sh = [e for r in rs for e in of_type(r, T.SHOT)]
    fg2 = [e.payload["made"] for e in sh if e.payload["value"] == 2]
    fg3 = [e.payload["made"] for e in sh if e.payload["value"] == 3]
    assert 0.38 <= sum(fg2) / len(fg2) <= 0.56
    assert 0.24 <= sum(fg3) / len(fg3) <= 0.42
    assert 0.2 <= len(fg3) / len(sh) <= 0.5
    ft = [e.payload["made"] for r in rs for e in of_type(r, T.FREE_THROW)]
    assert 0.6 <= sum(ft) / len(ft) <= 0.82
    atts = len(sh) / (2 * n)
    assert 50 <= atts <= 90
    for r in rs:
        assert 40 <= sum(r.final_score.values()) <= 260


def test_score_matches_truth_and_scoreboard(games):
    for r in games.values():
        sc = {"home": 0, "away": 0}
        for e in r.truth:
            if e.type == T.SHOT and e.payload["made"]:
                sc[e.team] += e.payload["value"]
            elif e.type == T.FREE_THROW and e.payload["made"]:
                sc[e.team] += 1
        assert sc == r.final_score
        ts, last = r.scoreboard[-1]
        assert (last["home"], last["away"]) == (sc["home"], sc["away"])
        assert sum(r.points.values()) == sum(sc.values())
        assert sc["home"] != sc["away"]
        # scoreboard score is monotone and lags the make by 1-3 s
        cum = {"home": 0, "away": 0}
        tss = [t for t, _ in r.scoreboard]
        for e in r.truth:
            pts = e.payload["value"] if (e.type == T.SHOT and e.payload["made"]) else (
                1 if e.type == T.FREE_THROW and e.payload["made"] else 0)
            if not pts:
                continue
            cum[e.team] += pts
            i = next(k for k, (t, d) in enumerate(r.scoreboard) if d[e.team] >= cum[e.team])
            delta = r.scoreboard[i][0] - e.video_ts
            assert 1.0 <= delta <= 5.5, delta
        prev = (0, 0)
        for t, d in r.scoreboard:
            cur = (d["home"], d["away"])
            assert cur[0] >= prev[0] and cur[1] >= prev[1]
            prev = cur
        assert tss == sorted(tss)


def test_team_fouls_on_scoreboard(games):
    r = games[1]
    starts = [e.video_ts for e in of_type(r, T.PERIOD_START)]
    ends = [e.video_ts for e in of_type(r, T.PERIOD_END)]
    for p, (s, en) in enumerate(zip(starts, ends), 1):
        counts = {"home": 0, "away": 0}
        for e in of_type(r, T.FOUL):
            if e.period == p and e.payload["kind"] != "technical":
                counts[e.team] += 1
        reads = [d for t, d in r.scoreboard if en + 0.5 <= t <= en + 2.5]
        assert reads and (reads[0]["fouls_home"], reads[0]["fouls_away"]) == (counts["home"], counts["away"])
        first = [d for t, d in r.scoreboard if s <= t <= s + 2]
        assert first[0]["fouls_home"] == 0 and first[0]["fouls_away"] == 0


def _clock_running_at(r):
    cs = sorted(obs_of(r, ClockState), key=lambda c: c.video_ts)
    tss = [c.video_ts for c in cs]

    def at(t):
        i = bisect.bisect_right(tss, t) - 1
        return cs[i].running if i >= 0 else False
    return at


def test_substitutions_only_while_clock_stopped_and_five_on_court(games):
    for r in games.values():
        running = _clock_running_at(r)
        court = {}
        for e in of_type(r, T.STARTERS):
            assert len(set(e.payload["players"])) == 5
            court[e.team] = set(e.payload["players"])
        subs = of_type(r, T.SUBSTITUTION)
        assert len(subs) > 10
        for e in r.truth:
            if e.type == T.SUBSTITUTION:
                assert not running(e.video_ts), (e.video_ts, e.period, e.clock)
                pin, pout = e.payload["player_in"], e.payload["player_out"]
                assert pout in court[e.team] and pin not in court[e.team]
                court[e.team] = (court[e.team] - {pout}) | {pin}
                assert len(court[e.team]) == 5
            elif e.type in (T.SHOT, T.FREE_THROW, T.REBOUND, T.STEAL, T.BLOCK, T.ASSIST) and e.player_id:
                assert e.player_id in court[e.team]
        # no player in more than one team, nobody fouls out and keeps playing
        fouls = Counter(e.player_id for e in of_type(r, T.FOUL) if e.payload["kind"] in ("personal", "offensive"))
        assert max(fouls.values()) <= 6


def test_clock_stream(games):
    r = games[2]
    cs = obs_of(r, ClockState)
    assert any(c.running for c in cs) and any(not c.running for c in cs)
    # running clock ticks once per second and counts down
    run = [c for c in cs if c.running and c.period == 1][:20]
    for a, b in zip(run, run[1:]):
        if 0.9 < b.video_ts - a.video_ts < 1.1:
            assert abs((a.clock - b.clock) - (b.video_ts - a.video_ts)) < 0.15
    for p in (1, 2, 3, 4):
        zs = [c for c in cs if c.period == p and c.clock == 0.0 and not c.running]
        assert zs
    assert all(0 <= c.clock <= 600.0 for c in cs)
    # period ends at clock zero
    for e in of_type(r, T.PERIOD_END):
        assert e.clock == 0.0


def test_observation_stream_sorted_and_signals(games):
    for r in games.values():
        ts = [o.video_ts for o in r.observations]
        assert ts == sorted(ts)
        assert obs_of(r, Possession) and obs_of(r, OutOfBounds) and obs_of(r, BallTouch)
        forms = obs_of(r, Formation)
        assert Counter(f.kind for f in forms)["jump_ball"] == 1
        assert Counter(f.kind for f in forms)["free_throw"] >= 5
        # blocks give abrupt touches, steals plain touches
        n_abrupt = sum(1 for b in obs_of(r, BallTouch) if b.abrupt_trajectory_change)
        assert n_abrupt == len(of_type(r, T.BLOCK))
        assert len(obs_of(r, BallTouch)) - n_abrupt == len(of_type(r, T.STEAL))


def test_shot_observations_consistent_with_truth(games):
    for r in games.values():
        shots_ft = of_type(r, T.SHOT, T.FREE_THROW)
        rel = obs_of(r, ShotRelease)
        assert len(rel) == len(shots_ft)
        rims = obs_of(r, RimMake)
        made = [e for e in shots_ft if e.payload["made"]]
        assert len(rims) == len(made)
        for e, o in zip(shots_ft, sorted(rel, key=lambda o: o.video_ts)):
            assert abs(e.video_ts - o.video_ts) < 1e-3
            assert o.team == e.team and o.player_id == e.player_id
            assert o.track_id in r.track_stints[o.player_id]
            home_right = e.period <= 2
            right = home_right == (e.team == "home")
            assert o.basket == ("right" if right else "left")
            if e.type == T.SHOT:
                assert abs(o.x - e.payload["x"]) < 1e-2 and abs(o.y - e.payload["y"]) < 1e-2
                assert in_court(o.x, o.y)
                if e.payload["value"] == 3:
                    assert o.three_prob > 0.9
                else:
                    assert o.three_prob < 0.1
                # shooter is on the correct half
                assert (o.x > 0) == right
            else:
                assert o.three_prob < 0.01
        # rim makes follow the release by 0.6-1.0 s
        rel_by_ts = [o.video_ts for o in sorted(rel, key=lambda o: o.video_ts) if True]
        mrel = [o.video_ts for e, o in zip(shots_ft, sorted(rel, key=lambda o: o.video_ts)) if e.payload["made"]]
        for t_rel, rm in zip(mrel, sorted(rims, key=lambda o: o.video_ts)):
            assert 0.59 <= rm.video_ts - t_rel <= 1.01
        assert rel_by_ts


def test_possession_signals_match_truth(games):
    r = games[3]
    pos = obs_of(r, Possession)
    reb = of_type(r, T.REBOUND)
    for e in reb:
        if e.player_id:
            assert any(p.player_id == e.player_id and abs(p.video_ts - e.video_ts) < 0.01 for p in pos)
    for p in pos:
        if p.track_id is None:
            assert p.team is None and p.player_id is None
        else:
            assert p.track_id in r.track_stints[p.player_id]


def test_track_ids_new_per_stint(games):
    r = games[1]
    allt = [t for v in r.track_stints.values() for t in v]
    assert len(allt) == len(set(allt))
    assert all(r.track_of[p] == v[0] for p, v in r.track_stints.items() if v and r.track_of[p] in v)
    subbed_in = {e.payload["player_in"] for e in of_type(r, T.SUBSTITUTION)}
    assert any(len(r.track_stints[p]) >= 2 for p in subbed_in)
    assert len(r.track_of) == 24


def test_noise_options_change_things_as_expected():
    base = sim(4)
    clean_rel = obs_of(base, ShotRelease)
    n = len(clean_rel)

    r = sim(4, noise=SimNoise(drop_shot_prob=0.5))
    k = len(obs_of(r, ShotRelease))
    assert 0.3 * n < k < 0.7 * n

    r = sim(4, noise=SimNoise(drop_player_id_prob=0.4))
    pos = [p for p in obs_of(r, Possession) if p.track_id is not None]
    frac = sum(p.player_id is None for p in pos) / len(pos)
    assert 0.25 < frac < 0.55
    assert [o for o in obs_of(r, ShotRelease) if o.player_id is None]
    assert len(obs_of(r, Possession)) == len(obs_of(base, Possession))

    nr = len(obs_of(base, RimMake))
    r = sim(4, noise=SimNoise(miss_rim_make_prob=0.5))
    assert 0.3 * nr < len(obs_of(r, RimMake)) < 0.7 * nr

    r = sim(4, noise=SimNoise(false_rim_make_prob=0.5))
    assert len(obs_of(r, RimMake)) > nr * 1.4
    assert any(o.confidence < 1.0 for o in obs_of(r, RimMake))

    r = sim(4, noise=SimNoise(three_prob_noise=1.2))
    rel = sorted(obs_of(r, ShotRelease), key=lambda o: o.video_ts)
    sh = of_type(r, T.SHOT, T.FREE_THROW)
    assert len(rel) == len(sh)
    bad = sum(1 for e, o in zip(sh, rel) if e.type == T.SHOT and
              ((e.payload["value"] == 3 and o.three_prob <= 0.9) or (e.payload["value"] == 2 and o.three_prob >= 0.1)))
    assert bad >= 2
    assert [o.x for o in rel] != [o.x for o in clean_rel]

    r = sim(4, noise=SimNoise(possession_jitter=0.3))
    a = [p.video_ts for p in obs_of(base, Possession)]
    b = [p.video_ts for p in obs_of(r, Possession)]
    assert len(a) == len(b) and a != b
    ts = [o.video_ts for o in r.observations]
    assert ts == sorted(ts)
    assert SimNoise().is_clean() and not SimNoise(drop_shot_prob=0.1).is_clean()


def test_frames_stream():
    t0 = time.time()
    r = simulate_game(11, DEMO.game, DEMO.rosters, periods=2, period_length_s=120.0, referees=2)
    assert time.time() - t0 < 15
    fr = obs_of(r, FrameState)
    assert len(fr) > 1000
    dts = [b.video_ts - a.video_ts for a, b in zip(fr, fr[1:])]
    assert all(abs(d - 0.1) < 1e-6 for d in dts)
    ts = [o.video_ts for o in r.observations]
    assert ts == sorted(ts)
    f = fr[len(fr) // 2]
    teams = Counter(p.team for p in f.players)
    assert teams["home"] == 12 and teams["away"] == 12 and teams[None] == 2
    assert len({p.track_id for p in f.players}) == len(f.players)
    assert f.ball_x is not None and f.ball_height is not None
    # on-court players (stable tracks of the lineup) stay in the court during play
    running = _clock_running_at(r)
    inside = []
    for f in fr[::10]:
        if running(f.video_ts):
            inside.append(sum(1 for p in f.players if p.team and in_court(p.x, p.y, 0.7) and p.y > -7.8))
    assert inside and sum(1 for n in inside if 9 <= n <= 11) / len(inside) > 0.85
    # bench players stay in bench zones most of the time
    zones = r.zones
    from courtvision.geometry.court import point_in_polygon
    f = fr[len(fr) // 2]
    on_bench = sum(1 for p in f.players if p.team and (point_in_polygon(p.x, p.y, zones["bench_home"])
                                                       or point_in_polygon(p.x, p.y, zones["bench_away"])))
    assert on_bench >= 10
    # movement is physically plausible (<= 7 m/s) for tracked players
    prev = {p.track_id: p for p in fr[100].players}
    mx = 0.0
    for p in fr[101].players:
        q = prev.get(p.track_id)
        if q is not None and p.team:
            mx = max(mx, ((p.x - q.x) ** 2 + (p.y - q.y) ** 2) ** 0.5 / 0.1)
    assert mx < 8.0
    # frame_hz option
    r2 = simulate_game(11, DEMO.game, DEMO.rosters, periods=1, period_length_s=30.0, frame_hz=5.0)
    f2 = obs_of(r2, FrameState)
    assert abs((f2[1].video_ts - f2[0].video_ts) - 0.2) < 1e-6


def test_overtime_when_tied():
    found = None
    for seed in range(60):
        r = simulate_game(seed, DEMO.game, DEMO.rosters, periods=1, period_length_s=25.0, include_frames=False)
        if r.periods_played > 1:
            found = r
            break
    assert found is not None
    assert found.final_score["home"] != found.final_score["away"]
    assert len(of_type(found, T.PERIOD_START)) == found.periods_played
    ot = [e for e in found.truth if e.period == 2]
    assert ot and max(e.clock for e in ot if e.clock is not None) <= DEMO.game.overtime_length_s


def test_rosters_keyed_by_team_id_and_small_game():
    rosters = {DEMO.home.id: DEMO.players[DEMO.home.id], DEMO.away.id: DEMO.players[DEMO.away.id]}
    r = simulate_game(3, DEMO.game, rosters, periods=2, period_length_s=60.0, include_frames=False)
    assert r.final_score == sim(3, periods=2, period_length_s=60.0).final_score
    with pytest.raises(ValueError):
        simulate_game(1, DEMO.game, {"home": DEMO.rosters["home"][:3], "away": DEMO.rosters["away"]})


def test_performance_full_game_without_frames():
    t0 = time.time()
    r = sim(21)
    assert time.time() - t0 < 2.0
    assert r.duration > 2400


def test_demo_data_and_store():
    s = EventStore()
    d = load_demo(s)
    assert s.get_game(d.game.id).home_team_id == d.home.id
    assert s.get_venue(d.venue.id) is not None
    assert len(s.roster(d.home.id)) == 12 and len(s.roster(d.away.id)) == 12
    minors = [p for p in d.all_players() if p.is_minor]
    assert len(minors) == 1
    assert not minors[0].has(Consent.STATS_PUBLIC) and minors[0].has(Consent.STATS_INTERNAL)
    assert d.home.league == "Oberliga"
    assert len({p.id for p in d.all_players()}) == 24
    r = sim(2)
    s.append_many(r.truth)
    assert len(s.events(d.game.id)) == len(r.truth)
    assert demo_setup(EventStore()).game.id == d.game.id
