from datetime import date, timedelta

import pytest

from courtvision.domain.events import Event, EventStatus, EventType as T, correct, void
from courtvision.domain.models import Game, Player
from courtvision.stats import (
    LeagueDistribution, PlayerLine, compute_boxscore, context_adjustment, eff, game_score,
    gmsc_per40, grade_boxscore, grade_player, leaderboard, pir, season_profile, to_grade,
)

GAME = Game(id="g", home_team_id="H", away_team_id="A")
HOME = [Player(id=f"h{i}", team_id="H", number=i, name=f"Home {i}") for i in range(1, 7)]
AWAY = [Player(id=f"a{i}", team_id="A", number=i, name=f"Away {i}") for i in range(1, 6)]
ROSTERS = {"home": HOME, "away": AWAY}


class Log:
    """Kleiner Event-Builder mit steigendem video_ts."""

    def __init__(self):
        self.events: list[Event] = []
        self._ts = 0.0

    def add(self, type_, period=1, clock=None, team=None, player=None, conf=1.0, status=None, **payload):
        self._ts += 1
        ev = Event(game_id="g", type=type_, video_ts=self._ts, period=period, clock=clock, team=team,
                   player_id=player, payload=payload, confidence=conf,
                   status=status or EventStatus.PROVISIONAL)
        self.events.append(ev)
        return ev


def build_game() -> Log:
    L = Log()
    L.add(T.STARTERS, clock=600, team="home", players=[f"h{i}" for i in range(1, 6)])
    L.add(T.STARTERS, clock=600, team="away", players=[f"a{i}" for i in range(1, 6)])
    L.add(T.PASS, clock=580, team="home", player="h1", to_id="h2")
    L.add(T.POSSESSION, clock=580, team="home", player="h1", start_ts=10.0, end_ts=14.5)
    s1 = L.add(T.SHOT, clock=570, team="home", player="h1", value=2, made=True, x=1.0, y=2.0)
    L.add(T.ASSIST, clock=570, team="home", player="h2", shot_id=s1.id)
    s2 = L.add(T.SHOT, clock=560, team="away", player="a1", value=3, made=False, x=8.0, y=0.0)
    L.add(T.BLOCK, clock=560, team="home", player="h3", conf=0.6, shot_id=s2.id, shooter_id="a1")
    L.add(T.REBOUND, clock=559, team="home", player="h4", offensive=False)
    L.add(T.SHOT, clock=540, team="away", player="a2", value=3, made=True, x=7.0, y=1.0)
    L.add(T.FOUL, clock=500, team="away", player="a3", kind="personal", fouled_id="h1")
    L.add(T.FREE_THROW, clock=500, team="home", player="h1", made=True, attempt=1, of=2)
    L.add(T.FREE_THROW, clock=500, team="home", player="h1", made=False, attempt=2, of=2)
    L.add(T.REBOUND, clock=499, team="away", player=None, offensive=False)
    L.add(T.SUBSTITUTION, clock=400, team="home", player_in="h6", player_out="h5")
    L.add(T.SHOT, clock=380, team="home", player="h6", value=3, made=True, x=7.0, y=-1.0)
    L.add(T.TURNOVER, clock=300, team="away", player="a1", conf=0.5, kind="bad_pass")
    L.add(T.TURNOVER, clock=290, team="home", player=None, kind="shot_clock")
    L.add(T.STEAL, clock=200, team="home", player="h2", victim_id="a3")
    L.add(T.FOUL, clock=200, team="away", player="a4", kind="technical", fouled_id=None)
    bad = L.add(T.SHOT, clock=150, team="home", player="h3", value=3, made=True, x=3.0, y=3.0)
    L.events.append(correct(bad, payload={"value": 2}))
    gone = L.add(T.SHOT, clock=120, team="away", player="a3", value=3, made=True, x=7.0, y=0.0)
    L.events.append(void(gone))
    L.add(T.SHOT, clock=100, team="home", player="h1", value=2, made=False, x=2.0, y=0.0)
    L.add(T.PERIOD_END, clock=0)
    L.add(T.PERIOD_START, period=2, clock=600)
    L.add(T.SHOT, period=2, clock=590, team="away", player="a5", value=2, made=True, x=1.0, y=0.0)
    L.add(T.GAME_END, period=2, clock=0)
    return L


@pytest.fixture
def box():
    return compute_boxscore(build_game().events, GAME, ROSTERS)


def test_player_counting_stats(box):
    h1 = box.players["h1"]
    assert (h1.pts, h1.fgm, h1.fga, h1.fg2m, h1.fg2a) == (3, 1, 2, 1, 2)
    assert (h1.ftm, h1.fta, h1.fd) == (1, 2, 1)
    assert (h1.passes, h1.possession_s) == (1, 4.5)
    assert h1.shots[0]["value"] == 2 and h1.shots[0]["made"] and len(h1.shots) == 2
    assert h1.efg_pct == pytest.approx(0.5)
    assert h1.ts_pct == pytest.approx(3 / (2 * (2 + 0.44 * 2)))
    assert box.players["h2"].ast == 1 and box.players["h2"].stl == 1
    h3 = box.players["h3"]
    assert h3.blk == 1 and (h3.pts, h3.fg2m, h3.fg3a) == (2, 1, 0)  # Korrektur 3 -> 2 gilt
    assert box.players["h4"].drb == 1 and box.players["h4"].reb == 1
    h6 = box.players["h6"]
    assert (h6.pts, h6.fg3m, h6.fg3a) == (3, 1, 1)
    a1 = box.players["a1"]
    assert (a1.fga, a1.fgm, a1.blka, a1.tov) == (1, 0, 1, 1)
    assert box.players["a1"].efg_pct == 0.0
    assert box.players["a3"].pf == 1 and box.players["a3"].fga == 0  # VOID-Wurf zählt nicht
    assert box.players["a4"].pf == 1
    assert box.players["a5"].pts == 2
    assert box.players["h5"].ts_pct is None and box.players["h5"].efg_pct is None


def test_team_lines(box):
    h, a = box.home, box.away
    assert h.points == 8 and h.totals.pts == 8
    assert (h.totals.fga, h.totals.fgm, h.totals.fg3a, h.totals.fta, h.totals.ftm) == (4, 3, 1, 2, 1)
    assert (h.totals.reb, h.totals.ast, h.totals.stl, h.totals.blk, h.totals.fd) == (1, 1, 1, 1, 1)
    assert h.team_turnovers == 1 and h.totals.tov == 1 and h.team_rebounds == 0
    assert a.points == 5 and a.totals.pts == 5
    assert (a.totals.fga, a.totals.fgm, a.totals.fg3a, a.totals.fg3m) == (3, 2, 2, 1)
    assert a.team_rebounds == 1 and a.totals.reb == 1 and a.totals.drb == 1
    assert a.totals.pf == 2 and a.totals.blka == 1 and a.totals.tov == 1
    # technisches Foul zählt nicht als Teamfoul
    assert a.team_fouls_by_period == {1: 1} and h.team_fouls_by_period == {}


def test_minutes_plus_minus(box):
    secs = {p: box.players[p].seconds_played for p in box.players}
    assert secs["h1"] == secs["h2"] == secs["h3"] == secs["h4"] == 1200
    assert secs["h5"] == 200 and secs["h6"] == 1000
    assert all(secs[f"a{i}"] == 1200 for i in range(1, 6))
    pm = {p: box.players[p].plus_minus for p in box.players}
    assert pm["h1"] == pm["h2"] == pm["h3"] == pm["h4"] == pm["h6"] == 3
    assert pm["h5"] == 0
    assert all(pm[f"a{i}"] == -3 for i in range(1, 6))


def test_lineups(box):
    by = {(l.side, l.players): l for l in box.lineups}
    assert len(by) == 3
    a = by[("home", ("h1", "h2", "h3", "h4", "h5"))]
    b = by[("home", ("h1", "h2", "h3", "h4", "h6"))]
    w = by[("away", ("a1", "a2", "a3", "a4", "a5"))]
    assert (a.seconds, a.pts_for, a.pts_against, a.net) == (200, 3, 3, 0)
    assert (b.seconds, b.pts_for, b.pts_against, b.net) == (1000, 5, 2, 3)
    assert (w.seconds, w.pts_for, w.pts_against, w.net) == (1200, 5, 8, -3)


def test_score_and_status(box):
    assert box.score == {"home": 8, "away": 5} and not box.score_mismatch
    assert box.status == "provisional"
    assert box.players["a1"].provisional_events == 2  # TO (0.5) + Block (0.6)
    assert box.players["h3"].provisional_events == 1
    assert box.players["h1"].provisional_events == 0
    L = build_game()
    L.add(T.GAME_FINALIZED, period=0, by="x")
    assert compute_boxscore(L.events, GAME, ROSTERS).status == "final"


def test_scoreboard_and_scoresheet():
    L = build_game()
    L.add(T.SCOREBOARD, period=2, clock=0, home=8, away=6, running=False, fouls_home=None, fouls_away=None)
    b = compute_boxscore(L.events, GAME, ROSTERS)
    assert b.score == {"home": 8, "away": 6} and b.score_mismatch
    L.add(T.SCOREBOARD, period=2, clock=0, home=8, away=5, running=False, fouls_home=0, fouls_away=0)
    b = compute_boxscore(L.events, GAME, ROSTERS)
    assert b.score == {"home": 8, "away": 5} and not b.score_mismatch  # letztes SCOREBOARD gilt
    L.add(T.SCORESHEET, period=0, players={}, home=9, away=5)
    b = compute_boxscore(L.events, GAME, ROSTERS)
    assert b.score == {"home": 9, "away": 5} and b.score_mismatch


def test_ft_after_sub_credited_to_lineup_on_court_at_scoring_event():
    L = Log()
    L.add(T.STARTERS, clock=600, team="home", players=[f"h{i}" for i in range(1, 6)])
    L.add(T.STARTERS, clock=600, team="away", players=[f"a{i}" for i in range(1, 6)])
    L.add(T.SUBSTITUTION, clock=300, team="home", player_in="h6", player_out="h5")
    L.add(T.FREE_THROW, clock=300, team="home", player="h1", made=True, attempt=1, of=1)
    b = compute_boxscore(L.events, GAME, ROSTERS)  # kein Periodenende -> Zeit bis zum letzten Event
    assert b.players["h6"].plus_minus == 1 and b.players["h5"].plus_minus == 0
    assert b.players["h5"].seconds_played == 300 and b.players["h6"].seconds_played == 0


def test_events_without_time_still_count():
    L = Log()
    L.add(T.SHOT, period=0, clock=None, team="home", player="h1", value=3, made=True, x=0, y=0)
    b = compute_boxscore(L.events, GAME, ROSTERS)
    assert b.players["h1"].pts == 3 and b.players["h1"].seconds_played == 0 and b.score["home"] == 3


# ------------------------------------------------------------------ Ratings
def mk_line(**kw) -> PlayerLine:
    base = dict(player_id="x", team="home", pts=20, fgm=8, fga=15, ftm=3, fta=4, orb=2, drb=5, reb=7,
                ast=4, stl=2, blk=1, blka=1, tov=3, pf=2, fd=5, seconds_played=1800)
    base.update(kw)
    return PlayerLine(**base)


def test_pir_eff_gmsc():
    ln = mk_line()
    assert pir(ln) == (20 + 7 + 4 + 2 + 1 + 5) - 7 - 1 - 3 - 1 - 2 == 25
    assert eff(ln) == 34 - 7 - 1 - 3 == 23
    assert game_score(ln) == pytest.approx(16.9)


def test_shrinkage():
    ln = mk_line()
    assert gmsc_per40(ln, 12.0) == pytest.approx((16.9 + 2.4) / 0.95)
    short = mk_line(pts=5, fgm=2, fga=2, ftm=1, fta=1, orb=0, drb=0, reb=0, ast=0, stl=0, blk=0, blka=0,
                    tov=0, pf=0, fd=0, seconds_played=120)
    raw = game_score(short) * 40 / 2
    s = gmsc_per40(short, 12.0)
    assert s == pytest.approx((game_score(short) + 2.4) / 0.25)
    assert 12.0 < s < raw  # zur Liga-Mitte gezogen
    assert gmsc_per40(mk_line(seconds_played=0, pts=0, fgm=0, fga=0, ftm=0, fta=0, orb=0, drb=0, ast=0,
                              stl=0, blk=0, tov=0, pf=0), 12.0) == pytest.approx(12.0)


def test_grade_mapping():
    assert to_grade(0.5) == 6.0 and to_grade(0) == 1.0 and to_grade(1) == 10.0
    assert to_grade(0.25) == 3.5 and to_grade(0.75) == 8.0 and to_grade(0.123) == 2.2


def test_context_adjustment_clamped():
    assert context_adjustment(100) == 0.5 and context_adjustment(-100) == -0.5
    assert context_adjustment(10) == pytest.approx(0.2)
    assert context_adjustment(0, 1.0) == pytest.approx(0.3)
    assert context_adjustment(0, 0.0) == pytest.approx(-0.3)
    assert context_adjustment(0, None) == 0
    assert context_adjustment(100, 1.0) == 0.5
    assert context_adjustment(10, pm_weight=0.1, cap=0.25) == 0.25


def test_distribution():
    d0 = date(2026, 6, 1)
    dist = LeagueDistribution(samples=[(d0, float(i)) for i in range(1, 26)])
    assert dist.percentile(13.0) == pytest.approx(0.5)
    assert dist.percentile(0.0) == 0.0 and dist.percentile(100.0) == 1.0
    dist.add(d0 - timedelta(days=500), 1000.0)  # älter als 12 Monate -> ignoriert
    assert dist.size() == 25 and dist.percentile(13.0) == pytest.approx(0.5)
    assert LeagueDistribution().percentile(5) == 0.5


def test_grade_player_fallback_and_distribution():
    zero = mk_line(seconds_played=0, plus_minus=0, pts=0, fgm=0, fga=0, ftm=0, fta=0, orb=0, drb=0, reb=0,
                   ast=0, stl=0, blk=0, blka=0, tov=0, pf=0, fd=0)
    g = grade_player(zero, None)  # Fallback: per40 == Ligamittel 12 -> Median
    assert g.value == 6.0 and g.per40 == pytest.approx(12.0)
    star = grade_player(mk_line(), LeagueDistribution())
    assert star.value > 6.0 and star.raw_gmsc == pytest.approx(16.9) and star.pir == 25 and star.eff == 23
    d0 = date(2026, 6, 1)
    dist = LeagueDistribution(samples=[(d0, float(i)) for i in range(1, 26)])
    top = grade_player(mk_line(), dist, provisional=True)  # per40 ~ 20 -> Perzentil 0.78
    assert top.provisional and 7.5 <= top.value <= 10 and 0.7 < top.percentile < 0.9
    bad = grade_player(mk_line(pts=0, fgm=0, ftm=0, tov=10), dist)
    assert bad.value < 3.0 and 1.0 <= bad.value
    plus = grade_player(mk_line(plus_minus=10), dist)
    assert plus.adjustment == pytest.approx(0.02 * 10 * 2400 / 1800)


def test_grade_boxscore(box):
    grades = grade_boxscore(box, None)
    assert set(grades) == set(box.players)  # alle mit Einsatzzeit
    assert all(1.0 <= g.value <= 10.0 for g in grades.values())
    assert grades["a1"].provisional and grades["h1"].provisional  # Box provisorisch
    L = build_game()
    L.add(T.GAME_FINALIZED, period=0, by="x")
    fin = compute_boxscore(L.events, GAME, ROSTERS)
    gf = grade_boxscore(fin, {"default": LeagueDistribution()})
    assert not gf["h1"].provisional and gf["a1"].provisional and gf["h3"].provisional
    # Spieler ohne Einsatzzeit bekommen keine Note
    assert "h6" in gf
    empty = compute_boxscore([], GAME, ROSTERS)
    assert grade_boxscore(empty) == {}


def test_season_profile():
    l1 = mk_line()
    l2 = mk_line(pts=10, fgm=4, fga=10, ftm=2, fta=0, reb=5, ast=2, stl=0, blk=0, tov=1)
    p = season_profile([l1, l2], [6.0, 7.0, 8.0, 5.0, 9.0, 6.5, 7.5], player_id="x")
    assert p.games == 2 and p.average_grade == pytest.approx(7.0, abs=0.01)
    assert p.form == [8.0, 5.0, 9.0, 6.5, 7.5]
    assert p.bests == {"pts": 20.0, "reb": 7.0, "ast": 4.0, "grade": 9.0}
    assert p.radar["scoring"] == pytest.approx(20 / 30 * 100, abs=0.1)
    assert p.radar["rebounding"] == pytest.approx(8 / 14 * 100, abs=0.1)
    assert p.radar["playmaking"] == pytest.approx(40.0)
    assert p.radar["defense"] == pytest.approx(40.0)
    assert p.radar["ball_security"] == pytest.approx(50.0)
    ts = 30 / (2 * (25 + 0.44 * 4))
    assert p.radar["efficiency"] == pytest.approx((ts - 0.35) / 0.35 * 100, abs=0.1)
    assert all(0 <= v <= 100 for v in p.radar.values())
    assert season_profile([], []).radar["scoring"] == 0


def test_leaderboard_excludes_minors():
    from courtvision.stats import Profile
    ps = [Profile(player_id="adult1", games=3, average_grade=7.0),
          Profile(player_id="kid", games=3, average_grade=9.5),
          Profile(player_id="adult2", games=3, average_grade=8.0),
          Profile(player_id="rookie", games=0, average_grade=10.0)]
    board = leaderboard(ps, minors={"kid"})
    assert [p.player_id for p in board] == ["adult2", "adult1"]
    assert [p.player_id for p in leaderboard(ps, minors=set(), limit=1, min_games=1)] == ["kid"]


def test_partial_scoreboard_reading_is_ignored():
    from courtvision.domain.events import Event, EventType
    from courtvision.domain.models import Game
    from courtvision.stats.boxscore import compute_boxscore
    g = Game(id="g", home_team_id="h", away_team_id="a")
    evs = [Event(game_id="g", type=EventType.SCOREBOARD, video_ts=1, payload={"home": 2, "away": 0}),
           Event(game_id="g", type=EventType.SCOREBOARD, video_ts=2, payload={"home": None, "away": 3})]
    assert compute_boxscore(evs, g, {"home": [], "away": []}).score == {"home": 2, "away": 0}


def test_offensive_foul_counts_as_turnover_once():
    from courtvision.domain.events import Event, EventType
    from courtvision.domain.models import Game
    from courtvision.stats.boxscore import compute_boxscore
    g = Game(id="g", home_team_id="h", away_team_id="a")
    foul = Event(game_id="g", type=EventType.FOUL, video_ts=5, team="home", player_id="p1",
                 payload={"kind": "offensive", "fouled_id": None})
    to = Event(game_id="g", type=EventType.TURNOVER, video_ts=5.5, team="home", player_id="p1",
               payload={"kind": "offensive_foul"})
    assert compute_boxscore([foul], g, {"home": [], "away": []}).players["p1"].tov == 1
    assert compute_boxscore([foul, to], g, {"home": [], "away": []}).players["p1"].tov == 1
