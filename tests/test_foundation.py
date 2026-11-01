import math
from courtvision.geometry.court import (is_three, three_probability, Calibration, LANDMARKS,
    BASKET_X, three_point_line, point_in_polygon, in_lane)
from courtvision.domain.events import Event, EventType, effective_events, confirm, void, game_seconds
from courtvision.domain.models import Game, Player, Team, Consent
from courtvision.store.eventstore import EventStore


def test_three_point_geometry():
    assert not is_three(BASKET_X - 6.0, 0, "right")
    assert is_three(BASKET_X - 7.0, 0, "right")
    assert is_three(-BASKET_X + 7.0, 0, "left")
    assert is_three(13.5, 6.9, "right")      # corner three
    assert not is_three(13.5, 6.3, "right")  # corner two
    assert not is_three(BASKET_X - 6.75, 0, "right")  # on the line = 2
    assert abs(three_probability(BASKET_X - 6.75, 0, "right") - 0.5) < 1e-9
    for x, y in three_point_line("right"):
        d = min(abs(abs(y) - 6.6), abs(math.hypot(x - BASKET_X, y) - 6.75))
        assert d < 1e-6
    assert in_lane(10, 0, "right") and not in_lane(-10, 0, "right")


def test_homography_roundtrip():
    # synthetic perspective camera
    import numpy as np
    H = np.array([[60, 10, 1900], [2, 25, 900], [0.001, 0.01, 1]])
    def proj(x, y):
        p = H @ [x, y, 1]; return (p[0] / p[2], p[1] / p[2])
    clicks = {n: proj(*c) for n, c in LANDMARKS.items() if n.startswith("corner") or n == "center"}
    cal = Calibration.from_landmarks(clicks)
    x, y = cal.to_court(*proj(3.3, -2.1))
    assert abs(x - 3.3) < 1e-6 and abs(y + 2.1) < 1e-6
    assert cal.reprojection_error() < 1e-6
    assert point_in_polygon(0.5, 0.5, [(0, 0), (1, 0), (1, 1), (0, 1)])


def test_event_sourcing_and_store():
    s = EventStore()
    s.put_team(Team(id="t1", name="A"))
    s.put_player(Player(id="p1", team_id="t1", number=7, name="X"))
    s.put_game(Game(id="g", home_team_id="t1", away_team_id="t2"))
    seen = []
    s.subscribe(seen.append)
    a = s.append(Event(game_id="g", type=EventType.SHOT, video_ts=5, payload={"value": 2, "made": True, "x": 0, "y": 0}))
    b = s.append(Event(game_id="g", type=EventType.REBOUND, video_ts=3, payload={"offensive": False}))
    c = s.append(confirm(a, payload={"value": 3}))
    eff = s.effective("g")
    assert [e.type for e in eff] == [EventType.REBOUND, EventType.SHOT]
    assert eff[1].payload["value"] == 3 and eff[1].payload["made"] is True
    s.append(void(c))
    assert [e.id for e in s.effective("g")] == [b.id]
    assert [e.seq for e in s.events("g")] == [1, 2, 3, 4] and len(seen) == 4
    assert s.roster("t1")[0].number == 7
    p = Player(id="p", team_id="t", number=1, name="n", consent=int(Consent.OBJECTED | Consent.HIGHLIGHTS))
    assert not p.has(Consent.HIGHLIGHTS)
    assert game_seconds(2, 300) == 900 and game_seconds(5, 0) == 2700
