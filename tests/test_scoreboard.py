import socket
import time

import numpy as np
import pytest

from courtvision.domain.events import Event, EventSource, EventStatus, EventType, effective_events
from courtvision.scoreboard import (DAKTRONICS_RTD_GENERIC, CameraScoreboardSource, FieldLayout,
                                    FieldReader, FusionConfig, PlausibilityFilter,
                                    ScoreboardFusion, ScoreboardLayout, ScoreboardOCR,
                                    ScoreboardReading, SegmentClassifier, TemplateClassifier,
                                    UdpScoreboardSource, render_digit, render_scoreboard_image,
                                    select_source)
from courtvision.scoreboard.sources import ScoreboardSource


# ------------------------------------------------------------------ OCR
@pytest.fixture(scope="module")
def template_clf():
    return TemplateClassifier.train_synthetic(n_per_class=250, seed=1)


@pytest.mark.parametrize("noise,blur", [(0.0, 0.0), (0.1, 0.5), (0.2, 0.8), (0.3, 1.0)])
@pytest.mark.parametrize("which", ["segment", "template"])
def test_digit_accuracy(noise, blur, which, template_clf):
    clf = SegmentClassifier() if which == "segment" else template_clf
    rng = np.random.default_rng(42)
    ok = n = 0
    for _ in range(40):
        for d in [None, *range(10)]:
            img = render_digit(d, (40, 24), noise=noise, blur=blur, rng=rng,
                               on_level=rng.uniform(0.75, 1.0), off_level=rng.uniform(0, 0.15))
            ok += clf.classify(img).digit == d
            n += 1
    assert ok / n >= 0.99, (which, noise, ok / n)


def test_digit_sizes_and_thickness():
    rng = np.random.default_rng(3)
    clf = SegmentClassifier()
    for size in [(30, 18), (60, 36), (80, 40)]:
        for t in (0.12, 0.25):
            for d in range(10):
                assert clf.classify(render_digit(d, size, t, noise=0.1, rng=rng)).digit == d


def test_flicker_and_partial_lit_averaging():
    rng = np.random.default_rng(5)
    clf = SegmentClassifier()
    layout = FieldLayout(roi=(0, 0, 48, 40), n_slots=2)
    def frame():
        a = render_digit(4, (40, 24), flicker=0.0, noise=0.05, rng=rng)
        b = render_digit(7, (40, 24), noise=0.05, rng=rng)
        img = np.hstack([a, b])
        # Flimmern: einzelne Frames mit stark gedimmten Segmenten
        if rng.random() < 0.4:
            img = img * rng.uniform(0.3, 0.5)
        return img
    fr = FieldReader(layout, clf, k=6)
    for _ in range(6):
        res = fr.read(frame())
    assert res.value == 47
    # Teilweise leuchtende Segmente werden toleriert
    ok = sum(clf.classify(render_digit(d, noise=0.05, dim_prob=0.0, flicker=0.3, rng=rng)).digit == d
             for d in range(10) for _ in range(10))
    assert ok >= 99


def _layout():
    def f(x, y, n, fmt="int", w=24):
        return FieldLayout(roi=(x, y, w * n, 40), n_slots=n, fmt=fmt)
    return ScoreboardLayout({
        "home": f(10, 10, 3), "away": f(120, 10, 3), "clock": f(10, 70, 4, "mm:ss"),
        "period": f(120, 70, 1), "fouls_home": f(10, 130, 2), "fouls_away": f(120, 130, 2),
    })


@pytest.mark.parametrize("which", ["segment", "template"])
def test_full_layout_read(which, template_clf):
    rng = np.random.default_rng(7)
    ocr = ScoreboardOCR(_layout(), classifier=None if which == "segment" else template_clf, k=3)
    texts = {"home": " 67", "away": "102", "clock": "0745", "period": "3", "fouls_home": " 4", "fouls_away": " 0"}
    r = None
    for i in range(3):
        img = render_scoreboard_image(ocr.layout, texts, (180, 220), rng, noise=0.08, blur=0.5, flicker=0.2)
        r = ocr.read(img, video_ts=float(i))
    assert (r.home, r.away, r.clock, r.period, r.fouls_home, r.fouls_away) == (67, 102, 465.0, 3, 4, 0)
    assert r.overall_confidence > 0.4
    ev = r.to_event("g")
    assert ev.type == EventType.SCOREBOARD and ev.source == EventSource.SCOREBOARD
    assert ev.status == EventStatus.CONFIRMED and ev.payload["home"] == 67
    cs = r.to_clock_state()
    assert cs.period == 3 and cs.clock == 465.0


def test_seconds_tenths_and_polarizer_hint():
    rng = np.random.default_rng(1)
    lay = ScoreboardLayout({
        "clock": FieldLayout((10, 10, 96, 40), 4, "mm:ss"),  # passt nicht zu "SS.t"
        "clock_alt": FieldLayout((10, 10, 72, 40), 3, "ss.t"),
    })
    ocr = ScoreboardOCR(lay, k=1)
    img = render_scoreboard_image(lay, {"clock_alt": "42" + "7"}, (60, 120), rng, noise=0.03)
    r = ocr.read(img)
    assert r.clock == pytest.approx(42.7)
    assert ocr.hints == []
    glare = render_scoreboard_image(lay, {"clock_alt": "427"}, (60, 120), rng, noise=0.0, glare=True)
    ocr.read(glare)
    assert ocr.hints and "Polfilter" in ocr.hints[0]


# --------------------------------------------------------- Plausibilität
def R(ts, **kw):
    return ScoreboardReading(video_ts=ts, **kw)


def feed(f, readings):
    out = []
    for r in readings:
        out.append(f.update(r))
    return out


def test_plausibility_rejects_glitch_until_stable():
    f = PlausibilityFilter(n_stable=3)
    outs = feed(f, [R(i, home=10, away=8, period=1, clock=500 - i) for i in range(4)])
    assert outs[-1].reading.home == 10
    # +4 Ausreisser (1 Frame) wird verworfen
    o = f.update(R(4, home=14, away=8, period=1, clock=496))
    assert o.reading.home == 10 and any(a.kind == "score_jump" and not a.accepted for a in o.anomalies)
    f.update(R(5, home=10, away=8, period=1, clock=495))
    # +2 stabil über 3 Frames -> übernommen
    outs = feed(f, [R(6 + i, home=12, away=8, period=1, clock=494 - i) for i in range(3)])
    assert outs[1].reading.home == 10 and outs[2].reading.home == 12


def test_plausibility_clock_and_period():
    f = PlausibilityFilter(n_stable=2)
    feed(f, [R(i, home=0, away=0, period=1, clock=300 - i) for i in range(3)])
    o = f.update(R(3, home=0, away=0, period=1, clock=400))  # Uhr springt hoch
    assert o.reading.clock == 298 and o.anomalies[0].kind == "clock_increase"
    o = f.update(R(4, home=0, away=0, period=1, clock=297))
    assert o.reading.clock == 297
    # Viertel +2 wird nicht übernommen
    outs = feed(f, [R(5 + i, home=0, away=0, period=3, clock=290) for i in range(3)])
    assert outs[-1].reading.period == 1
    assert any(a.kind == "period_jump" for o in outs for a in o.anomalies)
    # Viertel +1: Uhr darf zurückgesetzt werden, Teamfouls ebenfalls
    f.update(R(10, fouls_home=3, fouls_away=2))
    f.update(R(11, fouls_home=3, fouls_away=2))
    outs = feed(f, [R(12 + i, home=0, away=0, period=2, clock=600, fouls_home=0, fouls_away=0)
                    for i in range(3)])
    last = outs[-1].reading
    assert (last.period, last.clock, last.fouls_home) == (2, 600, 0)
    assert not [a for o in outs for a in o.anomalies if a.accepted]


# --------------------------------------------------------------- Quellen
def test_fixed_field_parser():
    pkt = b"\x01" + b" 67102" + b" 7:45" + b"3" + b" 4" + b" 1" + b"R" + b"\x04"
    r = DAKTRONICS_RTD_GENERIC.parse(pkt, 12.0)
    assert (r.home, r.away, r.clock, r.period, r.fouls_home, r.fouls_away) == (67, 102, 465.0, 3, 4, 1)
    assert r.running is True and not DAKTRONICS_RTD_GENERIC.verified
    pkt2 = b"\x01" + b"  0  1" + b"42.7 " + b"4" + b" 0" + b" 0" + b"S" + b"\x04"
    r2 = DAKTRONICS_RTD_GENERIC.parse(pkt2)
    assert r2.clock == pytest.approx(42.7) and r2.running is False
    buf = bytearray(b"junk" + pkt + pkt2[:5])
    assert DAKTRONICS_RTD_GENERIC.split(buf) == [pkt]
    buf += pkt2[5:]
    assert DAKTRONICS_RTD_GENERIC.split(buf) == [pkt2]


def test_udp_loopback():
    pkt = b"\x01" + b" 67102" + b" 7:45" + b"3" + b" 4" + b" 1" + b"R" + b"\x04"
    src = UdpScoreboardSource(DAKTRONICS_RTD_GENERIC, host="127.0.0.1", port=0)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(pkt, ("127.0.0.1", src.bound_port))
        s.close()
        got = []
        for _ in range(50):
            got = src.poll()
            if got:
                break
            time.sleep(0.01)
        assert got and got[0].home == 67 and src.healthy()
    finally:
        src.close()


class FakeSrc(ScoreboardSource):
    def __init__(self, quality, **kw):
        super().__init__(**kw)
        self.quality = quality

    def _poll(self):
        return [R(0, home=1)]


def test_source_selection_fallback():
    t = [0.0]
    clock = lambda: t[0]
    a, b, c = (FakeSrc(q, clock=clock, max_age=3.0) for q in (3, 2, 1))
    assert select_source([a, b, c]) is None
    for s in (a, b, c):
        s.poll()
    assert select_source([c, b, a]) is a
    t[0] = 2.0
    b.poll(); c.poll()
    t[0] = 4.0  # A veraltet
    assert select_source([a, b, c]) is b
    t[0] = 10.0
    assert select_source([a, b, c]) is None
    c.poll()
    assert select_source([a, b, c]) is c


def test_camera_source():
    rng = np.random.default_rng(0)
    ocr = ScoreboardOCR(_layout(), k=1)
    src = CameraScoreboardSource(ocr, "B", clock=lambda: 1.0)
    texts = {"home": "  5", "away": " 12", "clock": "0930", "period": "1", "fouls_home": " 1", "fouls_away": " 2"}
    img = render_scoreboard_image(ocr.layout, texts, (180, 220), rng, noise=0.05)
    r = src.push_frame(img, 1.0)
    assert r is not None and (r.home, r.away, r.clock, r.period) == (5, 12, 570.0, 1)
    assert r.source == src.name
    assert [x.home for x in src.poll()] == [5]
    assert select_source([src]) is src
    assert src.quality == 2


# ---------------------------------------------------------------- Fusion
def shot(ts, team="home", value=2, made=True, **kw):
    return Event(game_id="g", type=EventType.SHOT, video_ts=ts, team=team,
                 payload={"value": value, "made": made, "x": 5.0, "y": 1.0}, confidence=0.7, **kw)


def ft(ts, team="home", made=True):
    return Event(game_id="g", type=EventType.FREE_THROW, video_ts=ts, team=team,
                 payload={"made": made, "attempt": 1, "of": 2}, confidence=0.7)


def board(ts, home=0, away=0, **kw):
    kw.setdefault("period", 1)
    kw.setdefault("clock", 600.0 - ts)
    kw.setdefault("running", True)
    return R(ts, home=home, away=away, **kw)


def mk():
    fu = ScoreboardFusion("g")
    out = fu.on_reading(board(0))
    assert [e.type for e in out] == [EventType.SCOREBOARD]
    return fu


def non_sb(evs):
    return [e for e in evs if e.type != EventType.SCOREBOARD]


def test_match_make_confirms():
    fu = mk()
    s = shot(20.0)
    fu.on_optical_event(s)
    evs = non_sb(fu.on_reading(board(23, home=2)))
    assert len(evs) == 1
    e = evs[0]
    assert e.supersedes == s.id and e.status == EventStatus.CONFIRMED
    assert e.source == EventSource.SYSTEM and e.confidence >= 0.95
    assert e.period == 1 and e.clock == pytest.approx(580.0)
    assert fu.flush(60.0) == []  # nichts läuft ab


def test_fix_2_to_3():
    fu = mk()
    s = shot(20.0, value=2)
    fu.on_optical_event(s)
    e = non_sb(fu.on_reading(board(22, home=3)))[0]
    assert e.supersedes == s.id and e.payload["value"] == 3 and e.payload["made"] is True
    fu2 = mk()
    s = shot(20.0, value=3)
    fu2.on_optical_event(s)
    e = non_sb(fu2.on_reading(board(22, home=2)))[0]
    assert e.payload["value"] == 2


def test_missed_make_found_backwards():
    fu = mk()
    miss_far = shot(10.0, value=2, made=False)
    miss_near = shot(18.0, value=3, made=False)
    other_team = shot(19.5, team="away", value=2, made=False)
    for e in (miss_far, miss_near, other_team):
        fu.on_optical_event(e)
    e = non_sb(fu.on_reading(board(20, home=2)))
    assert len(e) == 1
    # Wert passt zu miss_far (2), nähere Alternative hat Wert 3 -> Wert hat Vorrang
    assert e[0].supersedes == miss_far.id
    assert e[0].payload["made"] is True and e[0].payload["value"] == 2
    assert e[0].confidence < 0.5 and e[0].status == EventStatus.PROVISIONAL


def test_new_shot_created():
    fu = mk()
    e = non_sb(fu.on_reading(board(30, away=3)))
    assert len(e) == 1 and e[0].type == EventType.SHOT and e[0].supersedes is None
    assert e[0].team == "away" and e[0].player_id is None
    assert e[0].payload["value"] == 3 and e[0].payload["made"] is True
    assert e[0].confidence == pytest.approx(0.3)


def test_optical_make_without_board_change_becomes_miss():
    fu = mk()
    s = shot(20.0)
    fu.on_optical_event(s)
    fu.on_reading(board(25))
    assert fu.flush(28.0) == []  # Fenster noch offen
    evs = non_sb(fu.on_reading(board(31))) + fu.flush(32.0)
    assert len(evs) == 1 and evs[0].supersedes == s.id and evs[0].payload["made"] is False
    # Tafel ausgefallen: keine Umwandlung
    fu2 = mk()
    fu2.on_optical_event(shot(20.0))
    assert fu2.flush(100.0) == []


def test_free_throw_plus_one():
    fu = mk()
    f = ft(40.0)
    fu.on_optical_event(f)
    e = non_sb(fu.on_reading(board(42, away=0, home=1)))
    assert len(e) == 1 and e[0].supersedes == f.id and e[0].type == EventType.FREE_THROW
    assert e[0].status == EventStatus.CONFIRMED
    # Fehlender FT -> neuer FT
    e = non_sb(fu.on_reading(board(60, home=2)))
    assert e[0].type == EventType.FREE_THROW and e[0].payload["made"] is True


def test_and_one_and_multi_jump():
    fu = mk()
    s, f = shot(20.0, value=2), ft(21.0)
    fu.on_optical_event(s); fu.on_optical_event(f)
    evs = non_sb(fu.on_reading(board(23, home=3)))
    assert {e.supersedes for e in evs if e.type != EventType.FLAG} == {s.id, f.id}
    evs = non_sb(fu.on_reading(board(50, home=8)))  # +5 ohne Optik
    kinds = [e.payload.get("kind") for e in evs if e.type == EventType.FLAG]
    assert "multi_point_jump" in kinds
    assert sum(e.payload.get("value", 0) for e in evs if e.type == EventType.SHOT) == 5


def test_foul_candidate_and_period_reset():
    fu = mk()
    fu.on_reading(board(10, fouls_home=2, fouls_away=0))
    evs = non_sb(fu.on_reading(board(20, fouls_home=3, fouls_away=0)))
    assert len(evs) == 1 and evs[0].type == EventType.FLAG
    assert evs[0].payload["kind"] == "foul_candidate" and evs[0].team == "home"
    # neues Viertel: Rücksetzen löst keinen Kandidaten aus
    evs = non_sb(fu.on_reading(board(30, period=2, clock=600.0, fouls_home=0, fouls_away=0)))
    assert evs == []


def test_scoreboard_event_rate_limit():
    fu = mk()
    n = sum(e.type == EventType.SCOREBOARD for ts in range(1, 5) for e in fu.on_reading(board(ts)))
    assert n == 0  # nur Uhr geändert, < 5 s
    out = fu.on_reading(board(7))
    assert any(e.type == EventType.SCOREBOARD for e in out)


def test_late_optical_replaces_synthetic():
    fu = mk()
    created = non_sb(fu.on_reading(board(30, home=2)))[0]
    s = shot(29.0)
    evs = fu.on_optical_event(s)
    assert evs[0].type == EventType.VOID and evs[0].supersedes == created.id
    assert evs[1].supersedes == s.id


def test_reconcile_with_scoresheet():
    fu = mk()
    fu.on_reading(board(10, home=2))
    assert fu.reconcile_with_scoresheet((2, 0), 2, 0) == []
    evs = fu.reconcile_with_scoresheet(fu._state, 3, 0)
    assert len(evs) == 1 and evs[0].payload["kind"] == "scoreboard_mismatch"
    assert evs[0].payload["winner"] == "scoresheet"


def test_fusion_ft_series_with_lagging_board_matches_fifo():
    """3 Freiwürfe im Abstand ~3 s, Tafel zählt jeweils ~3.5 s verspätet hoch."""
    from courtvision.domain.events import Event, EventType
    from courtvision.scoreboard import ScoreboardFusion, ScoreboardReading
    fus = ScoreboardFusion("g")
    out = []
    def board(ts, away):
        out.extend(fus.on_reading(ScoreboardReading(video_ts=ts, home=10, away=away, clock=45.0,
                                                    period=3, running=False, fouls_home=2, fouls_away=1)))
    fts = []
    board(0.0, 20)
    for i, t in enumerate((2.0, 5.2, 8.6)):
        ev = Event(game_id="g", type=EventType.FREE_THROW, video_ts=t, team="away", player_id="p",
                   payload={"made": True, "attempt": i + 1, "of": 3})
        fts.append(ev)
        out.extend(fus.on_optical_event(ev))
    for ts in range(1, 20):
        away = 20 + sum(1 for t in (5.5, 8.7, 12.1) if ts >= t)
        board(float(ts), away)
    out.extend(fus.flush(20.0))
    from courtvision.domain.events import effective_events
    eff = [e for e in effective_events(fts + out) if e.type == EventType.FREE_THROW]
    assert len(eff) == 3 and all(e.payload["made"] for e in eff)
    assert all(e.player_id == "p" for e in eff)
