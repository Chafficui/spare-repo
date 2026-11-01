"""Synthetische Tests fuer courtvision.cv (keine ML-Modelle noetig)."""
from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from courtvision.cv import (BallSample, CVPipeline, Detection, FormationDetector, NumberVoter,
                            OutOfBoundsDetector, PipelineConfig, PossessionEstimator, RimConfig,
                            RimMakeDetector, RosterAssigner, RosterEntry, ShotDetector, TeamAssigner, Tracker,
                            TrackerConfig, TrackVotes, color_histogram_embedding, crop_regions, downscale,
                            fit_parabola, foot_point, hand_points, heatmap_peak, kmeans)
from courtvision.cv.detection import RFDETRDetector, DetectorUnavailableError, rim_roi
from courtvision.cv.teams import rgb_identity_embedding, parse_color
from courtvision.domain.observations import (BallTouch, ClockState, Formation, FrameState, OutOfBounds, PlayerState,
                                             Possession, RimMake, ShotRelease)
from courtvision.geometry.court import Calibration

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- Detektion
def test_downscale_and_crops():
    frame = np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)
    ds = downscale(frame, factor=2)
    assert ds.image.shape == (40, 60, 3) and ds.scale_x == 2.0
    ds2 = downscale(frame, max_side=60)  # exakt Faktor 2
    assert ds2.image.shape[1] == 60
    ds3 = downscale(frame, max_side=50)  # nicht ganzzahlig -> nearest
    assert max(ds3.image.shape[:2]) == 50 and ds3.scale_x > 2
    assert downscale(frame, max_side=500).image is frame

    crops = crop_regions(frame, scale=2.0, player_boxes={7: (10, 5, 20, 35)}, ball_candidates=[(30, 20)],
                         rim_rois={"left": rim_roi((10, 10), 4)})
    kinds = sorted(c.kind for c in crops)
    assert kinds == ["ball", "jersey", "rim"]
    j = next(c for c in crops if c.kind == "jersey")
    assert j.track_id == 7 and j.box[0] <= 20 and j.box[2] >= 40  # Volbild-Koordinaten
    assert j.image.shape[0] == j.box[3] - j.box[1]
    rim = next(c for c in crops if c.kind == "rim")
    assert rim.name == "left" and rim.image.size > 0


def test_rfdetr_adapter_clear_error_without_package():
    pytest.importorskip  # noqa: B018
    try:
        import rfdetr  # noqa: F401
        pytest.skip("rfdetr installiert")
    except ImportError:
        pass
    with pytest.raises(DetectorUnavailableError, match="rfdetr"):
        RFDETRDetector(size="nano")
    with pytest.raises(ValueError):
        RFDETRDetector(size="huge", lazy=True)
    assert RFDETRDetector(size="small", checkpoint_path="x.pth", lazy=True).size == "small"


def test_detection_validation():
    with pytest.raises(ValueError):
        Detection((0, 0, 1, 1), 0.5, "coach")
    d = Detection((10, 20, 30, 60), 0.9, "player")
    assert d.foot_point == (20, 60)


# --------------------------------------------------------------------------- Tracking
def _box(cx, cy, w=40, h=100, noise=0.0):
    n = RNG.normal(0, noise, 4) if noise else np.zeros(4)
    return (cx - w / 2 + n[0], cy - h / 2 + n[1], cx + w / 2 + n[2], cy + h / 2 + n[3])


def _run(tracker, trajs, n, skip=None, embs=None, noise=1.0):
    """trajs: Liste Funktionen t->(cx,cy). Gibt je Objekt die Folge der Track-IDs zurueck."""
    ids = [[] for _ in trajs]
    for t in range(n):
        dets, owner = [], []
        for k, f in enumerate(trajs):
            if skip and skip(k, t):
                continue
            dets.append(Detection(_box(*f(t), noise=noise), 0.9))
            owner.append(k)
        out = tracker.update(dets, None if embs is None else [embs[k] for k in owner])
        for tr in out:
            ids[owner[tr.det_index]].append(tr.id)
    return ids


def test_tracker_crossing_no_id_switch():
    a = lambda t: (100 + 8 * t, 200)  # noqa: E731
    b = lambda t: (500 - 7 * t, 230)  # noqa: E731
    ids = _run(Tracker(), [a, b], 45)
    for seq in ids:
        assert len(seq) >= 40 and len(set(seq)) == 1
    assert ids[0][0] != ids[1][0]


def test_tracker_crossing_with_embeddings():
    e1, e2 = np.array([1.0, 0, 0.1]), np.array([0, 1.0, 0.1])
    a = lambda t: (100 + 6 * t, 200)  # noqa: E731
    b = lambda t: (400 - 6 * t, 205)  # noqa: E731  # fast vollstaendige Ueberdeckung beim Kreuzen
    ids = _run(Tracker(), [a, b], 50, embs=[e1, e2])
    assert all(len(set(s)) == 1 for s in ids) and ids[0][0] != ids[1][0]


def test_tracker_occlusion_gap_keeps_id_and_oru_recovers_state():
    traj = lambda t: (100 + 6 * t, 300)  # noqa: E731
    gap = range(20, 28)
    tr = Tracker(TrackerConfig(max_age=30))
    ids = _run(tr, [traj], 40, skip=lambda k, t: t in gap, noise=0.5)
    assert len(set(ids[0])) == 1  # selbe ID vor/nach Verdeckung
    trk = tr.active_tracks()[0]
    assert abs(trk.velocity[0] - 6) < 1.0  # Geschwindigkeit nach Re-Update plausibel
    cx = (trk.bbox[0] + trk.bbox[2]) / 2
    assert abs(cx - traj(39)[0]) < 5

    # ohne ORU und mit grosser Luecke ist die Geschwindigkeit schlechter oder gleich
    tr2 = Tracker(TrackerConfig(max_age=30, oru=False))
    _run(tr2, [traj], 40, skip=lambda k, t: t in gap, noise=0.5)
    assert abs(trk.velocity[0] - 6) <= abs(tr2.active_tracks()[0].velocity[0] - 6) + 0.5


def test_tracker_birth_and_death():
    tr = Tracker(TrackerConfig(min_hits=3, max_age=5))
    d = [Detection(_box(100, 100))]
    assert tr.update(d) == [] and tr.update(d) == []
    out = tr.update(d)
    assert len(out) == 1 and out[0].id == 1 and out[0].hits == 3
    for _ in range(5):
        assert tr.update([]) == []
    assert len(tr.tracks) == 1
    tr.update([])
    assert tr.tracks == []
    # Fehldetektion (1 Frame) wird nie zu einem Track
    tr.update([Detection(_box(300, 300))])
    for _ in range(4):
        tr.update([])
    assert tr.tracks == []


def test_tracker_embedding_transient():
    tr = Tracker()
    for _ in range(4):
        tr.update([Detection(_box(100, 100))], [np.array([1.0, 2.0])])
    t = tr.active_tracks()[0]
    assert t.embedding is not None and abs(np.linalg.norm(t.embedding) - 1) < 1e-9
    assert pickle.loads(pickle.dumps(t)).embedding is None  # nie mitserialisiert
    tr.clear_embeddings()
    assert t.embedding is None
    assert t.foot_point[1] == t.bbox[3]


# --------------------------------------------------------------------------- Teams
def _team_features(n_home=5, n_away=5, n_ref=2, noise=8.0):
    home, away, ref = (200, 30, 30), (30, 60, 200), (20, 20, 20)
    f = {}
    tid = 1
    for _ in range(n_home):
        f[tid] = np.array(home) + RNG.normal(0, noise, 3); tid += 1
    for _ in range(n_away):
        f[tid] = np.array(away) + RNG.normal(0, noise, 3); tid += 1
    for _ in range(n_ref):
        f[tid] = np.array(ref) + RNG.normal(0, noise, 3); tid += 1
    return f


def test_kmeans_split():
    X = np.vstack([RNG.normal(0, 0.3, (20, 2)), RNG.normal(5, 0.3, (20, 2))])
    lab, cen, _ = kmeans(X, 2)
    assert len(set(lab[:20])) == 1 and len(set(lab[20:])) == 1 and lab[0] != lab[-1]


def test_team_assigner_seeded_with_referees_outliers():
    f = _team_features()
    ta = TeamAssigner(seed_colors={"home": (200, 30, 30), "away": (30, 60, 200)}, seed_embed=rgb_identity_embedding)
    for t, v in f.items():
        ta.observe(t, v)
    assert ta.fit()
    assert [ta.team_of(t) for t in range(1, 6)] == ["home"] * 5
    assert [ta.team_of(t) for t in range(6, 11)] == ["away"] * 5
    assert ta.team_of(11) is None and ta.team_of(12) is None  # Schiris: dritter, ferner Cluster
    # Seed-Zuordnung: vertauschte Seeds -> vertauschte Labels
    tb = TeamAssigner(seed_colors={"home": (30, 60, 200), "away": (200, 30, 30)}, seed_embed=rgb_identity_embedding)
    for t, v in f.items():
        tb.observe(t, v)
    tb.fit()
    assert tb.team_of(1) == "away" and tb.team_of(6) == "home"
    ta.clear()
    assert ta.team_of(1) is None


def test_team_assigner_referee_by_class_and_unseeded():
    f = _team_features(n_ref=0)
    ta = TeamAssigner()
    for t, v in f.items():
        ta.observe(t, v)
    ta.observe(99, np.array([20.0, 20, 20]), is_referee=True)
    assert ta.fit() and not ta.seeded
    teams = [ta.team_of(t) for t in range(1, 11)]
    assert len(set(teams[:5])) == 1 and len(set(teams[5:])) == 1 and teams[0] != teams[5]
    assert ta.team_of(99) is None


def test_color_histogram_embedding_separates_colors_and_seed_colors():
    def shirt(rgb):
        img = np.full((100, 50, 3), 60, np.uint8)
        img[:, :] = np.clip(np.array(rgb) + RNG.normal(0, 6, (100, 50, 3)), 0, 255).astype(np.uint8)
        return img
    red, blue, black = (color_histogram_embedding(shirt(c)) for c in [(210, 30, 30), (30, 60, 210), (15, 15, 15)])
    red2 = color_histogram_embedding(shirt((200, 40, 35)))
    assert abs(red.sum() - 1) < 1e-9
    assert np.linalg.norm(red - red2) < 0.2 * np.linalg.norm(red - blue)
    assert np.linalg.norm(red - blue) > 0.5 and np.linalg.norm(blue - black) > 0.5
    ta = TeamAssigner(seed_colors={"home": parse_color("#d21e1e"), "away": parse_color("blue")})
    for i in range(4):
        ta.observe(i, color_histogram_embedding(shirt((210, 30, 30))))
        ta.observe(10 + i, color_histogram_embedding(shirt((30, 60, 210))))
    assert ta.fit() and ta.team_of(0) == "home" and ta.team_of(10) == "away"


# --------------------------------------------------------------------------- Trikotnummern
def test_number_voter_confirms_after_three_and_samples_every_5th():
    v = NumberVoter()
    sampled = [f for f in range(20) if v.should_sample(1, f)]
    assert sampled == [0, 5, 10, 15]
    assert v.add_reading(1, 23, 0.9) is None
    assert v.add_reading(1, 23, 0.8) is None
    assert v.add_reading(1, 8, 0.9) is None  # abweichend
    assert v.confirmed(1) is None
    assert v.add_reading(1, 23, 0.7) == 23
    assert v.votes(1).numbers[23] == pytest.approx(2.4)
    # schwache Lesungen zaehlen nicht zur Bestaetigung
    for _ in range(5):
        v.add_reading(2, 11, 0.2)
    assert v.confirmed(2) is None
    v.add_reading(3, None, 0.9)
    assert v.votes(3).unreadable == pytest.approx(0.9) and v.confirmed(3) is None


def _roster(prefix, nums, available=None):
    return [RosterEntry(f"{prefix}{n}", n, available is None or n in available) for n in nums]


def _votes(number, weight=3.0, extra=None):
    v = TrackVotes(numbers={number: weight}, counts={number: 3}, n_readings=3, confirmed=number)
    for n, w in (extra or {}).items():
        v.numbers[n] = w
    return v


def test_roster_assignment_no_duplicates_and_conflict_resolution():
    ra = RosterAssigner({"home": _roster("h", [4, 5, 6, 7, 8, 9, 10])})
    # Tracks 1 und 2 lesen beide "7"; Track 1 deutlich sicherer, Track 2 sieht auch 8
    res = ra.assign("home", {1: _votes(7, 6.0), 2: _votes(7, 3.0, {8: 2.5}), 3: _votes(5)}, clock_stopped=False)
    assert res[1].player_id == "h7" and res[2].player_id == "h8" and res[3].player_id == "h5"
    ids = [r.player_id for r in res.values() if r.player_id]
    assert len(ids) == len(set(ids))
    assert res[1].confidence > 0.8 and res[1].candidates[0][0] == "h7"


def test_roster_excludes_unavailable_and_unreadable_candidates():
    ra = RosterAssigner({"home": _roster("h", [4, 5, 6, 7, 8, 9], available={4, 5, 6, 7, 8})})
    res = ra.assign("home", {1: _votes(9), 2: _votes(4), 3: TrackVotes(unreadable=2.0, n_readings=2)}, False)
    assert res[1].player_id != "h9"  # nicht verfuegbar (z. B. disqualifiziert) -> nie zugeordnet
    assert res[2].player_id == "h4"
    assert res[3].player_id is None
    cand = {c for c, _ in res[3].candidates}
    assert "h4" not in cand and "h9" not in cand and {"h5", "h6", "h7", "h8"} <= cand


def test_roster_max_five_and_clock_stopped_rule():
    ra = RosterAssigner({"home": _roster("h", range(4, 13))})
    five = {i: _votes(3 + i) for i in range(1, 6)}  # Nummern 4..8
    res = ra.assign("home", five, clock_stopped=False, ts=0.0)
    assert {r.player_id for r in res.values()} == {f"h{n}" for n in range(4, 9)}
    # Track 6 liest 12, Track 5 (h8) verschwindet: Uhr laeuft -> fuenfter Feldplatz noch belegt
    t6 = {i: five[i] for i in range(1, 5)}
    t6[6] = _votes(12)
    res = ra.assign("home", t6, clock_stopped=False, ts=1.0)
    assert res[6].player_id is None  # max. 5 verletzt -> keine Zuordnung
    assert res[6].candidates
    # Uhr steht (Wechselfenster): h12 darf rein, h8 wird verdraengt
    res = ra.assign("home", t6, clock_stopped=True, ts=2.0)
    assert res[6].player_id == "h12"
    assert "h8" not in ra.on_court["home"] and len(ra.on_court["home"]) == 5


def test_roster_identity_change_only_when_clock_stopped_unless_high_confidence():
    ra = RosterAssigner({"home": _roster("h", [4, 5, 6])})
    assert ra.assign("home", {1: _votes(4)}, False)[1].player_id == "h4"
    flip = {1: TrackVotes(numbers={5: 3.0, 4: 1.0}, counts={5: 3, 4: 1}, n_readings=4, confirmed=5)}
    assert ra.assign("home", flip, clock_stopped=False)[1].player_id == "h4"  # bleibt
    assert ra.assign("home", flip, clock_stopped=True)[1].player_id == "h5"  # Wechselfenster
    ra2 = RosterAssigner({"home": _roster("h", [4, 5, 6])})
    ra2.assign("home", {1: _votes(4)}, False)
    sure = {1: TrackVotes(numbers={6: 400.0, 4: 3.0}, counts={6: 40}, n_readings=40, confirmed=6)}
    assert ra2.assign("home", sure, clock_stopped=False)[1].player_id == "h6"  # sehr hohe Konfidenz


# --------------------------------------------------------------------------- Ball
def test_fit_parabola():
    t = np.linspace(0, 1, 12)
    z = 2.0 + 6.0 * t - 4.905 * t**2 + RNG.normal(0, 0.005, t.size)
    p = fit_parabola(t, z)
    assert p.a == pytest.approx(-4.905, abs=0.2) and p.b == pytest.approx(6.0, abs=0.2)
    assert p.apex_t == pytest.approx(6 / 9.81, abs=0.05) and p.rmse < 0.02 and p(t[0]) == pytest.approx(2, abs=0.05)
    # Bildraum y(x)
    x = np.linspace(100, 400, 10)
    y = 0.002 * (x - 250) ** 2 + 80
    assert fit_parabola(x, y).a == pytest.approx(0.002, rel=1e-6)
    with pytest.raises(ValueError):
        fit_parabola([0, 1], [0, 1])


def test_heatmap_peak_subpixel():
    yy, xx = np.mgrid[0:48, 0:64]
    hm = np.exp(-((xx - 30.4) ** 2 + (yy - 20.7) ** 2) / (2 * 1.5**2))
    x, y, s = heatmap_peak(hm)
    assert x == pytest.approx(30.4, abs=0.15) and y == pytest.approx(20.7, abs=0.15) and s > 0.9
    assert heatmap_peak(hm * 0.2) is None


def _drive_shot(sd, foot, v, z0=2.2, hz=20, steps=14, team="home", defenders=(), reverse_at=None, release=1.0,
                mode="height"):
    """Spielt Halten (bis release) + Flug nach; liefert alle erzeugten Observations."""
    out = []
    holder = PlayerState(1, team, foot[0], foot[1], "p1")
    for i in range(int(release * hz)):
        out += sd.update(i / hz, BallSample(i / hz, foot[0], foot[1], 1.5, None), [holder, *defenders],
                         [Possession(0.0, 1, team, "p1")] if i == 0 else [])
    vx, vy, vz = v
    x, y = foot
    for k in range(steps):
        t = release + (k + 1) / hz
        d = (k + 1) / hz
        if reverse_at is not None and k >= reverse_at:
            vx_eff = -vx * 0.5
            base = (x + vx * (reverse_at / hz), y + vy * (reverse_at / hz))
            bx, by = base[0] + vx_eff * (d - reverse_at / hz), base[1]
        else:
            bx, by = x + vx * d, y + vy * d
        z = z0 + vz * d - 4.905 * d * d
        s = BallSample(t, bx, by, z if mode == "height" else None, None if mode == "height" else -40 * z)
        ev = [Possession(release, None, None)] if k == 1 else []
        out += sd.update(t, s, [holder, *defenders], ev)
    return out


def test_shot_detection_three_vs_two_and_basket():
    # hinter der Linie (7,4 m vom rechten Korb), Wurf Richtung rechter Korb
    out = _drive_shot(ShotDetector(), (5.0, 0.0), (6.0, 0.0, 6.5))
    shots = [o for o in out if isinstance(o, ShotRelease)]
    assert len(shots) == 1
    s = shots[0]
    assert s.basket == "right" and s.track_id == 1 and s.team == "home" and s.player_id == "p1"
    assert (s.x, s.y) == (5.0, 0.0) and s.video_ts == pytest.approx(1.0) and s.three_prob > 0.99
    # innerhalb der Linie
    s2 = [o for o in _drive_shot(ShotDetector(), (9.5, 2.0), (4.0, -0.7, 6.0)) if isinstance(o, ShotRelease)][0]
    assert s2.basket == "right" and s2.three_prob < 0.01
    # linker Korb, Ecke (hinter der Eckenlinie |y|>6.6)
    s3 = [o for o in _drive_shot(ShotDetector(), (-11.0, 7.2), (-3.0, -2.5, 6.0), team="away")
          if isinstance(o, ShotRelease)][0]
    assert s3.basket == "left" and s3.three_prob > 0.9 and s3.team == "away"
    # direkt auf der Linie: Unsicherheitsband
    s4 = [o for o in _drive_shot(ShotDetector(), (12.425 - 6.75, 0.0), (6.0, 0.0, 6.5)) if isinstance(o, ShotRelease)][0]
    assert 0.3 < s4.three_prob < 0.7


def test_shot_detection_image_mode_and_non_shots():
    out = _drive_shot(ShotDetector(), (5.0, 0.0), (6.0, 0.0, 6.5), mode="image")
    assert len([o for o in out if isinstance(o, ShotRelease)]) == 1
    # Pass: flach, keine Aufwaertsbahn
    assert not [o for o in _drive_shot(ShotDetector(), (5.0, 0.0), (9.0, 2.0, 0.3), z0=1.3) if isinstance(o, ShotRelease)]
    # Dribbling: Ball nach unten
    assert not [o for o in _drive_shot(ShotDetector(), (5.0, 0.0), (0.2, 0.0, -2.0), z0=1.0) if isinstance(o, ShotRelease)]
    # Aufwaertsbahn, aber quer zum Korb (zur Seitenlinie)
    assert not [o for o in _drive_shot(ShotDetector(), (5.0, 0.0), (0.3, 6.0, 6.5)) if isinstance(o, ShotRelease)]


def test_blocked_shot_emits_abrupt_ball_touch():
    defender = PlayerState(9, "away", 5.8, 0.1, "d9")
    out = _drive_shot(ShotDetector(), (5.0, 0.0), (6.0, 0.0, 6.5), defenders=[defender], reverse_at=3)
    assert len([o for o in out if isinstance(o, ShotRelease)]) == 1
    touches = [o for o in out if isinstance(o, BallTouch)]
    assert len(touches) == 1 and touches[0].abrupt_trajectory_change and touches[0].track_id == 9
    assert touches[0].team == "away"
    # ungehinderter Wurf: kein Touch
    assert not [o for o in _drive_shot(ShotDetector(), (5.0, 0.0), (6.0, 0.0, 6.5), defenders=[defender])
                if isinstance(o, BallTouch)]


def _rim_track(det, pts, dt=1 / 30, t0=0.0):
    fired = []
    for i, p in enumerate(pts):
        m = det.update(t0 + i * dt, p)
        if m:
            fired.append((t0 + i * dt, m))
    return fired


def test_rim_make_top_to_bottom_vs_bounce_out():
    mk = lambda: RimMakeDetector("right", (500.0, 300.0), 20.0)  # noqa: E731
    make = [(503, 230 + 9 * i) for i in range(14)]  # 230 -> 347, von oben durch den Ring
    fired = _rim_track(mk(), make)
    assert len(fired) == 1 and fired[0][1].basket == "right" and isinstance(fired[0][1], RimMake)
    # Netz verdeckt den Ball nach dem Durchgang
    fired = _rim_track(mk(), make[:9] + [None] * 8)
    assert len(fired) == 1
    # Abpraller: faellt auf den Ring, springt nach oben weg
    bounce = [(505, 230 + 9 * i) for i in range(8)] + [(510 + 6 * i, 290 - 10 * i) for i in range(8)]
    assert max(v for _, v in bounce) > 290 and _rim_track(mk(), bounce) == []
    # von unten nach oben durch die Ringebene
    assert _rim_track(mk(), list(reversed(make))) == []
    # neben dem Ring vorbei (ausserhalb des Radius)
    assert _rim_track(mk(), [(545, 230 + 9 * i) for i in range(14)]) == []
    # taucht ein, kommt aber wieder hoch (Rim-out)
    out = [(500, 230 + 9 * i) for i in range(9)] + [(500, 311 - 9 * i) for i in range(1, 8)]
    assert _rim_track(mk(), out) == []


# --------------------------------------------------------------------------- Besitz / Aus / Formation
def _ps(tid, team, x, y):
    return PlayerState(tid, team, x, y)


def test_possession_hysteresis_three_frames():
    pe = PossessionEstimator()
    pl = [_ps(1, "home", 0.0, 0.0), _ps(2, "away", 5.0, 5.0)]
    near, far = (0.4, 0.2), (3.0, 0.0)
    ev = [pe.update(i * 0.1, pl, near) for i in range(2)]
    assert ev == [None, None]  # nur 2 Frames
    assert pe.update(0.2, pl, far) is None  # Serie reisst ab
    got = [pe.update(1.0 + i * 0.1, pl, near) for i in range(3)]
    assert got[:2] == [None, None] and isinstance(got[2], Possession)
    assert got[2].track_id == 1 and got[2].team == "home" and got[2].video_ts == pytest.approx(1.0)
    assert pe.update(1.3, pl, near) is None  # nur bei Aenderung
    assert pe.update(1.4, pl, None) is None  # 1 Frame Ballverlust: gehalten
    assert pe.update(1.5, pl, near) is None
    assert pe.update(1.6, pl, far) is None
    free = pe.update(1.7, pl, far)
    assert free.track_id is None and free.video_ts == pytest.approx(1.6)
    assert pe.last_touch.track_id == 1
    # Hand-Keypoints haben Vorrang: Hand weit vom Ball -> kein Besitz
    pe2 = PossessionEstimator()
    for i in range(5):
        assert pe2.update(i * 0.1, pl, near, hands={1: [(2.0, 2.0)]}) is None
    ev = [pe2.update(1 + i * 0.1, pl, near, hands={1: [(0.5, 0.3)]}) for i in range(3)]
    assert isinstance(ev[-1], Possession)
    # Ball ueber Reichweite (Wurf): kein Besitz
    pe3 = PossessionEstimator()
    assert all(pe3.update(i * 0.1, pl, near, ball_height=4.5) is None for i in range(5))


def test_out_of_bounds():
    ob = OutOfBoundsDetector()
    from courtvision.cv.possession import LastTouch
    lt = LastTouch(3, "home", "h7", 5.0)
    assert ob.update(6.0, (14.4, 2.0), False, lt) is None  # draussen, aber Spiel laeuft
    assert ob.update(6.1, (14.8, 2.0), False, lt) is None
    res = ob.update(6.2, (15.2, 2.0), True, lt)
    assert isinstance(res, OutOfBounds) and res.last_touch_track == 3 and res.last_touch_player == "h7"
    assert res.video_ts == pytest.approx(6.0)
    assert ob.update(6.3, (15.5, 2.0), True, lt) is None  # nur einmal je Episode
    assert ob.update(6.4, (3.0, 2.0), True, lt) is None  # wieder drin
    assert ob.update(7.0, (3.0, 9.0), True, lt) is None  # 1 Frame reicht nicht
    assert isinstance(ob.update(7.1, (3.0, 9.1), True, lt), OutOfBounds)
    assert OutOfBoundsDetector().update(0, (0, 0), True) is None  # drin


def test_free_throw_and_jump_ball_formation():
    fd = FormationDetector()
    shooter = PlayerState(5, "home", 8.2 + 0.2, 0.1, "h5")
    lane = [_ps(10 + i, "home" if i % 2 else "away", 11.5 + (i // 2) * 1.0, 2.45 * (1 if i % 2 else -1))
            for i in range(4)]
    others = [_ps(20, "away", 4.0, 3.0)]
    res = [fd.update(i * 0.1, [shooter, *lane, *others]) for i in range(10)]
    got = [r for r in res if r]
    assert len(got) == 1 and isinstance(got[0], Formation)
    assert got[0].kind == "free_throw" and got[0].basket == "right"
    assert got[0].shooter_track == 5 and got[0].shooter_team == "home" and got[0].shooter_player == "h5"
    # zu wenig Spieler an den Zonenraendern: kein Freiwurf
    fd2 = FormationDetector()
    assert not any(fd2.update(i * 0.1, [shooter, *lane[:2]]) for i in range(10))
    # Schuetze zu weit von der Linie
    far = PlayerState(5, "home", 7.0, 0.0)
    assert not any(FormationDetector().update(i * 0.1, [far, *lane]) for i in range(10))
    # linker Korb (gespiegelt)
    fl = FormationDetector()
    m_shooter = PlayerState(5, "away", -8.2, 0.0)
    m_lane = [_ps(10 + i, "home", -11.0 - i * 0.5, 2.45 if i % 2 else -2.45) for i in range(4)]
    r = [fl.update(i * 0.1, [m_shooter, *m_lane]) for i in range(10)]
    assert [x.basket for x in r if x] == ["left"]
    # Sprungball
    fj = FormationDetector()
    jb = [_ps(1, "home", -0.3, 0.0), _ps(2, "away", 0.3, 0.1)] + [_ps(3 + i, "home" if i % 2 else "away", 3 + i, 2 - i) for i in range(6)]
    r = [fj.update(i * 0.1, jb) for i in range(10)]
    kinds = [x.kind for x in r if x]
    assert kinds == ["jump_ball"]
    # drei Spieler im Kreis -> kein Sprungball
    jb3 = jb + [_ps(30, "home", 0.0, 1.0)]
    assert not any(FormationDetector().update(i * 0.1, jb3) for i in range(10))


# --------------------------------------------------------------------------- Pose
def test_pose_helpers():
    kp = np.zeros((17, 3))
    kp[9] = (100, 100, 0.9)  # left_wrist
    kp[7] = (100, 60, 0.9)  # left_elbow
    kp[10] = (150, 120, 0.1)  # right_wrist, unsicher
    h = hand_points(kp)
    assert len(h) == 1 and h[0] == pytest.approx((100, 110))
    assert foot_point(kp) is None
    kp[15] = (50, 200, 0.9); kp[16] = (70, 210, 0.8)
    assert foot_point(kp) == pytest.approx((60, 205))
    kp[16, 2] = 0.0
    assert foot_point(kp) == pytest.approx((50, 200))


# --------------------------------------------------------------------------- Pipeline E2E
PX, PY, LIFT = 40.0, 25.0, 40.0  # Pixel je m (x, y) und je m Hoehe


def _court_to_px(x, y, z=0.0):
    return PX * x + 600.0, -PY * y + 400.0 - LIFT * z


def _calibration():
    from courtvision.geometry.court import LANDMARKS
    names = ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right", "center",
             "ft_left_top", "ft_right_bottom"]
    return Calibration.from_landmarks({n: _court_to_px(*LANDMARKS[n]) for n in names})


def _pdet(x, y, cls="player"):
    u, v = _court_to_px(x, y)
    return Detection((u - 15, v - 80, u + 15, v), 0.9, cls)


def _shirt(rgb):
    img = np.empty((80, 30, 3), np.uint8)
    img[:] = np.clip(np.array(rgb) + RNG.normal(0, 5, (80, 30, 3)), 0, 255).astype(np.uint8)
    return color_histogram_embedding(img)


def test_calibration_helper_roundtrip():
    cal = _calibration()
    assert cal.reprojection_error() < 1e-6
    x, y = cal.to_court(*_court_to_px(5.0, 2.0))
    assert (x, y) == pytest.approx((5.0, 2.0), abs=1e-6)


def test_pipeline_end_to_end_shot_and_make():
    cal = _calibration()
    home_pos = [(2.0, -4.0), (5.0, 2.0), (-3.0, 3.0), (-6.0, -2.0), (0.0, 6.0)]  # Index 1 = Werfer #7
    away_pos = [(8.0, -3.0), (-8.0, 4.0), (10.0, 5.0), (-2.0, -5.0), (3.0, -0.5)]
    ref_pos = [(0.0, -6.5)]
    home_nums, away_nums = [4, 7, 8, 9, 10], [4, 5, 6, 7, 8]
    rosters = {"home": [RosterEntry(f"h{n}", n) for n in [4, 7, 8, 9, 10, 11, 12]],
               "away": [RosterEntry(f"a{n}", n) for n in [4, 5, 6, 7, 8, 9]]}
    rim_u, rim_v = _court_to_px(12.425, 0.0, 3.05)
    pipe = CVPipeline(cal, PipelineConfig.edge(), seed_colors={"home": (200, 30, 30), "away": (30, 60, 200)},
                      rosters=rosters, rims={"right": RimConfig((rim_u, rim_v), 10.0)})
    pipe.set_clock(ClockState(0.0, 1, 600.0, True))
    emb_home = [_shirt((200, 30, 30)) for _ in home_pos]
    emb_away = [_shirt((30, 60, 200)) for _ in away_pos]
    emb_ref = _shirt((15, 15, 15))

    release, hz = 0.8, 30
    vx = 6.0
    T = (12.425 - 5.0) / vx
    vz = (3.05 - 2.0 + 4.905 * T * T) / T
    obs = []
    for i in range(int(3.0 * hz)):
        ts = i / hz
        dets, embs, reads = [], [], {}
        for k, (x, y) in enumerate(home_pos):
            dets.append(_pdet(x + RNG.normal(0, 0.01), y)); embs.append(emb_home[k])
            reads[len(dets) - 1] = (home_nums[k], 0.9)
        for k, (x, y) in enumerate(away_pos):
            dets.append(_pdet(x, y + RNG.normal(0, 0.01))); embs.append(emb_away[k])
            reads[len(dets) - 1] = (away_nums[k], 0.9)
        for x, y in ref_pos:
            dets.append(_pdet(x, y, "referee")); embs.append(emb_ref)
        if ts < release:
            u, v = _court_to_px(5.0, 2.0)
            bu, bv = u + 10, v - 55
        else:
            d = ts - release
            bu, bv = _court_to_px(5.0 + vx * d, 2.0 * (1 - d / T), 2.0 + vz * d - 4.905 * d * d)
        dets.append(Detection((bu - 5, bv - 5, bu + 5, bv + 5), 0.8, "ball")); embs.append(None)
        obs += pipe.process_tracks(ts, dets, embs, reads)
        if i == 0:
            assert not any(isinstance(o, FrameState) for o in obs)  # Tracks erst nach min_hits
    obs += pipe.end_game()

    ts_list = [o.video_ts for o in obs]
    assert ts_list == sorted(ts_list)  # geordneter Strom
    frames = [o for o in obs if isinstance(o, FrameState)]
    assert 25 <= len(frames) <= 31  # ~10 Hz ueber 3 s
    last = frames[-1]
    assert len(last.players) == 11
    teams = {p.track_id: p.team for p in last.players}
    assert sum(t == "home" for t in teams.values()) == 5 and sum(t == "away" for t in teams.values()) == 5
    assert sum(t is None for t in teams.values()) == 1  # Schiri
    shooter_state = min((p for p in last.players if p.team == "home"), key=lambda p: math.hypot(p.x - 5.0, p.y - 2.0))
    assert shooter_state.player_id == "h7" and shooter_state.id_confidence > 0.5
    assert math.hypot(shooter_state.x - 5.0, shooter_state.y - 2.0) < 0.2
    assert {p.player_id for p in last.players if p.team == "away"} == {f"a{n}" for n in away_nums}

    poss = [o for o in obs if isinstance(o, Possession)]
    assert poss[0].track_id == shooter_state.track_id and poss[0].team == "home"
    assert poss[-1].track_id is None
    shots = [o for o in obs if isinstance(o, ShotRelease)]
    assert len(shots) == 1
    s = shots[0]
    assert s.basket == "right" and s.team == "home" and s.track_id == shooter_state.track_id
    assert s.three_prob > 0.99 and release - 0.1 <= s.video_ts <= release + 0.6
    assert (s.x, s.y) == pytest.approx((5.0, 2.0), abs=0.2)
    makes = [o for o in obs if isinstance(o, RimMake)]
    assert len(makes) == 1 and makes[0].basket == "right"
    assert release + T - 0.1 <= makes[0].video_ts <= release + T + 0.1
    assert not [o for o in obs if isinstance(o, OutOfBounds)]  # Uhr laeuft
    # Datenschutz: nach Spielende keine Embeddings mehr
    assert all(t.embedding is None for t in pipe.tracker.tracks) and not pipe.teams.fitted


def test_pipeline_ignores_bench_and_requires_detector_for_frames():
    cal = _calibration()
    pipe = CVPipeline(cal, PipelineConfig.cloud())
    with pytest.raises(RuntimeError):
        pipe.process_frame(np.zeros((10, 10, 3), np.uint8), 0.0)
    obs = []
    for i in range(10):
        obs += pipe.process_tracks(i * 0.1, [_pdet(0.0, 0.0), _pdet(0.0, -12.0)])  # 2. Spieler weit ausserhalb
    obs += pipe.flush()
    fs = [o for o in obs if isinstance(o, FrameState)]
    assert fs and all(len(f.players) == 1 for f in fs[-3:])
    assert PipelineConfig.cloud().detect_max_side is None and PipelineConfig.edge().detect_max_side == 1280


def test_pipeline_process_frame_with_fake_detector():
    cal = _calibration()

    class Fake:
        def detect(self, frame):  # arbeitet im herunterskalierten Bild (Faktor 2)
            return [Detection((u / 2 - 7, v / 2 - 40, u / 2 + 7, v / 2), 0.9, "player")
                    for u, v in [_court_to_px(0.0, 0.0), _court_to_px(4.0, 1.0)]]

    pipe = CVPipeline(cal, PipelineConfig.edge(detect_max_side=1000), detector=Fake())
    frame = np.zeros((800, 2000, 3), np.uint8)  # Faktor 2 auf die lange Seite
    obs = []
    for i in range(6):
        obs += pipe.process_frame(frame, i * 0.1)
    obs += pipe.flush()
    fs = [o for o in obs if isinstance(o, FrameState)]
    assert fs and len(fs[-1].players) == 2
    xs = sorted(p.x for p in fs[-1].players)
    assert xs == pytest.approx([0.0, 4.0], abs=0.15)
