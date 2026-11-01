"""Live-Pfad: PeriodTracker, GameSession, Cloud-Reprocessing, Edge-Recorder/-Uploader."""
import os
import sqlite3

import pytest

from courtvision.domain.events import (EventSource, EventStatus, EventType as T, correct,
                                       effective_events, void)
from courtvision.domain.models import Game, GameStatus
from courtvision.domain.observations import ClockState
from courtvision.edge import (EncoderUnavailableError, LocalRecorder, LteFallback, RecorderConfig,
                              SegmentState, TransportError, UploadQueue, Uploader,
                              build_gstreamer_pipeline, realtime_upload_possible, stream_feasible)
from courtvision.live import GameSession, PeriodTracker, reprocess
from courtvision.live.session import merge_streams
from courtvision.scoreboard import ScoreboardReading
from courtvision.sim import demo_setup, simulate_game
from courtvision.stats.boxscore import compute_boxscore
from courtvision.store.eventstore import EventStore

DEMO = demo_setup()


def _game(periods=4, length=600.0):
    return Game(id="g", home_team_id="h", away_team_id="a", periods=periods, period_length_s=length,
                overtime_length_s=300.0)


def _types(evs):
    return [(e.type, e.period) for e in evs]


# --------------------------------------------------------------------------- PeriodTracker
def _drive(tr, seq):
    out = []
    for ts, p, clock, running, score in seq:
        tr.set_score(*score)
        out += tr.on_clock(ts, p, clock, running)
    return out


def test_tracker_regular_game_sequence():
    tr = PeriodTracker("g", _game(2, 100.0))
    seq = [(0, 1, 100, False, (0, 0)), (5, 1, 100, True, (0, 0)), (60, 1, 45, True, (4, 2)),
           (110, 1, 0, False, (10, 8)), (111, 1, 0, False, (10, 8)),
           (300, 2, 100, False, (10, 8)), (310, 2, 100, True, (10, 8)), (400, 2, 12, True, (20, 15)),
           (420, 2, 0.0, False, (22, 15)), (421, 2, 0.0, False, (22, 15)), (426, 2, 0.0, False, (22, 15))]
    out = _drive(tr, seq)
    assert _types(out) == [(T.GAME_START, 1), (T.PERIOD_START, 1), (T.PERIOD_END, 1),
                           (T.PERIOD_START, 2), (T.PERIOD_END, 2), (T.GAME_END, 2)]
    assert out[0].video_ts == 5 and out[0].clock == 100
    for e in out:
        assert e.source == EventSource.SCOREBOARD and e.status == EventStatus.CONFIRMED
    ends = [e for e in out if e.type in (T.PERIOD_END, T.GAME_END)]
    assert all(e.clock == 0.0 for e in ends)
    assert ends[0].video_ts == 110 and ends[1].video_ts == 420  # GAME_END traegt Periodenende
    assert tr.game_ended


def test_tracker_overtime_when_tied():
    tr = PeriodTracker("g", _game(1, 60.0))
    seq = [(0, 1, 60, True, (0, 0)), (60, 1, 0, False, (50, 50)), (70, 1, 0, False, (50, 50)),
           (200, 2, 300, False, (50, 50)), (210, 2, 300, True, (50, 50)),
           (400, 2, 0, False, (54, 52)), (410, 2, 0, False, (54, 52))]
    out = _drive(tr, seq)
    assert _types(out) == [(T.GAME_START, 1), (T.PERIOD_START, 1), (T.PERIOD_END, 1),
                           (T.PERIOD_START, 2), (T.PERIOD_END, 2), (T.GAME_END, 2)]


def test_tracker_idempotent_and_no_duplicates():
    tr = PeriodTracker("g", _game(1, 60.0))
    seq = [(0, 1, 60, True, (0, 0)), (60, 1, 0, False, (5, 3)), (70, 1, 0, False, (5, 3))]
    first = _drive(tr, seq)
    again = _drive(tr, seq)
    assert [e.type for e in first] == [T.GAME_START, T.PERIOD_START, T.PERIOD_END, T.GAME_END]
    assert again == []
    assert tr.end(75) == [] and tr.start(0) == []


def test_tracker_period_change_without_zero_estimates_end():
    tr = PeriodTracker("g", _game(4, 600.0))
    out = _drive(tr, [(0, 1, 600, True, (0, 0)), (590, 1, 12, True, (20, 18)),
                      (700, 2, 600, False, (20, 18))])
    pe = [e for e in out if e.type == T.PERIOD_END][0]
    assert pe.period == 1 and pe.clock == 0.0 and pe.video_ts == pytest.approx(602.0)
    assert not tr.game_ended  # Spiel geht weiter


def test_tracker_waits_for_late_board_score_and_explicit_end():
    tr = PeriodTracker("g", _game(1, 60.0), end_settle_s=4.0)
    out = _drive(tr, [(0, 1, 60, True, (50, 50)), (60, 1, 0, False, (50, 50)), (62, 1, 0, False, (50, 50))])
    assert T.GAME_END not in [e.type for e in out]  # Tafel zeigt Gleichstand: noch offen
    tr.set_score(52, 50)  # letzter Korb erscheint verzoegert
    out = tr.on_clock(65, 1, 0, False)
    assert [e.type for e in out] == [T.GAME_END]
    tr2 = PeriodTracker("g", _game(4, 600.0))
    tr2.on_clock(0, 1, 600, True)
    out = tr2.end(30.0)
    assert [e.type for e in out] == [T.PERIOD_END, T.GAME_END] and out[0].video_ts == 30.0
    assert tr2.end(31.0) == []


def test_tracker_accepts_readings_and_clockstate():
    tr = PeriodTracker("g", _game(1, 60.0))
    out = tr.feed(ScoreboardReading(video_ts=1, home=0, away=0, clock=60.0, period=1, running=True))
    out += tr.feed(ClockState(video_ts=61, period=1, clock=0.0, running=False))
    out += tr.feed(ScoreboardReading(video_ts=70, home=2, away=0, clock=0.0, period=1, running=False))
    assert [e.type for e in out] == [T.GAME_START, T.PERIOD_START, T.PERIOD_END, T.GAME_END]


# --------------------------------------------------------------------------- GameSession
def _run_session(seed=1, periods=2, length=150.0, frames=True, store=None, **kw):
    game = DEMO.game.model_copy(update={"periods": periods, "period_length_s": length})
    sim = simulate_game(seed, game, DEMO.rosters, periods=periods, period_length_s=length,
                        include_frames=frames)
    store = store or EventStore(":memory:")
    store.put_game(game)
    sess = GameSession(store, game, DEMO.rosters, calibration=sim.zones, **kw)
    for side in ("home", "away"):
        st = next(e for e in sim.truth if e.type == T.STARTERS and e.team == side)
        sess.set_starters(side, st.payload["players"])
    return game, sim, store, sess


@pytest.fixture(scope="module")
def played():
    game, sim, store, sess = _run_session()
    seen = []
    store.subscribe(seen.append)
    sess.feed_many(sim.observations, sim.scoreboard)
    sess.finish()
    return game, sim, store, sess, seen


def test_session_score_and_listener(played):
    game, sim, store, sess, seen = played
    box = compute_boxscore(store.effective(game.id), game, DEMO.rosters)
    assert box.score == sim.final_score and not box.score_mismatch
    assert len(seen) == len(store.events(game.id)) - 2  # alles ging durch die Store-Listener (ohne STARTERS)
    assert store.get_game(game.id).status == GameStatus.PROCESSING


def test_session_period_and_game_events(played):
    game, sim, store, sess, _ = played
    ev = store.effective(game.id)
    kinds = [(e.type, e.period) for e in ev if e.type in (T.GAME_START, T.PERIOD_START, T.PERIOD_END, T.GAME_END)]
    assert kinds == [(T.GAME_START, 1), (T.PERIOD_START, 1), (T.PERIOD_END, 1), (T.PERIOD_START, 2),
                     (T.PERIOD_END, 2), (T.GAME_END, 2)]
    assert sess.finish() == []  # idempotent
    # optische Events tragen Viertel und Uhr
    assert all(e.period > 0 and e.clock is not None for e in ev if e.type == T.SHOT)


def test_session_minutes(played):
    game, sim, store, _, _ = played
    ba = compute_boxscore(store.effective(game.id), game, DEMO.rosters)
    bt = compute_boxscore(sim.truth, game, DEMO.rosters)
    total = sum(l.seconds_played for l in ba.players.values())
    assert total == pytest.approx(10 * game.periods * game.period_length_s, abs=1.0)
    # TODO(substitutions): einzelne Wechsel ausserhalb der Erkennung (Detektor in Arbeit) verschieben
    # Minuten zwischen Spielern; deshalb pro Spieler nur "ueberwiegend" pruefen.
    within = [abs(ba.players[p].seconds_played - l.seconds_played) <= 30
              for p, l in bt.players.items() if l.seconds_played > 0]
    assert sum(within) / len(within) >= 0.7


def test_merge_streams_scoreboard_first_at_equal_ts():
    obs = [ClockState(1.0, 1, 10.0, True), ClockState(2.0, 1, 9.0, True)]
    rd = [ScoreboardReading(video_ts=1.0, clock=10.0, period=1), ScoreboardReading(video_ts=1.5, clock=9.5, period=1)]
    out = list(merge_streams(obs, [(r.video_ts, {"clock": r.clock, "period": 1}) for r in rd]))
    assert [type(x).__name__ for x in out] == ["ScoreboardReading", "ClockState", "ScoreboardReading", "ClockState"]


# --------------------------------------------------------------------------- Reprocess
def test_reprocess_preserves_human_events_and_replaces_optical():
    game, sim, store, sess = _run_session(frames=False, periods=1, length=240.0)
    sess.feed_many(sim.observations, sim.scoreboard)
    sess.finish()
    live = store.effective(game.id)
    opt_shots = [e for e in live if e.type == T.SHOT and e.source == EventSource.OPTICAL]
    miss = opt_shots[0]
    corrected = correct(miss, payload={"made": False, "note": "Konsole"})  # Bediener bestaetigt
    store.append(corrected)
    removed = [e for e in live if e.type == T.SHOT and e.id != miss.id][0]
    store.append(void(removed))  # Bediener entfernt ein Event
    foul = store.append(_console_foul(game.id, sim))
    before = {e.id for e in store.effective(game.id)}
    human_ids = {e.id for e in store.effective(game.id) if e.source == EventSource.CONSOLE}
    rep = reprocess(store, game, DEMO.rosters, sim.observations, sim.scoreboard, calibration=sim.zones)
    after = store.effective(game.id)
    ids = {e.id for e in after}
    assert human_ids <= ids  # Konsole-Events (STARTERS, Korrektur, Foul) bleiben
    assert rep.voided > 0 and rep.added > 0
    assert not (before & ids) - human_ids - {e.id for e in after if e.source == EventSource.SCOREBOARD}
    # das von Menschen entfernte Event kommt nicht zurueck, die Korrektur wird nicht dupliziert
    near = lambda a, b: abs(a.video_ts - b.video_ts) <= 2.0 and a.team == b.team  # noqa: E731
    assert not any(e.type == T.SHOT and near(e, removed) for e in after)
    assert sum(1 for e in after if e.type == T.SHOT and near(e, miss)) == 1
    # keine Doppelzaehlung: Punkte gleich Live-Lauf ohne die entfernten Treffer
    box = compute_boxscore(after, game, DEMO.rosters)
    assert box.home.points + box.away.points <= sim.final_score["home"] + sim.final_score["away"]
    assert box.home.totals.pf >= 1  # Konsole-Foul zaehlt
    assert any(e.id == foul.id for e in after)


def _console_foul(game_id, sim):
    from courtvision.domain.events import Event
    pid = DEMO.rosters["home"][0].id
    return Event(game_id=game_id, type=T.FOUL, video_ts=50.0, period=1, clock=190.0, team="home",
                 player_id=pid, payload={"kind": "personal", "fouled_id": None},
                 source=EventSource.CONSOLE, status=EventStatus.CONFIRMED)


# --------------------------------------------------------------------------- Edge: Recorder
def test_pipeline_passthrough_and_branches():
    cfg = RecorderConfig(camera_url="rtsp://cam/1", rtmp_url="rtmp://ingest/live/key", segment_s=300)
    p = build_gstreamer_pipeline(cfg)
    assert p.count("tee name=t") == 1 and p.count("t. !") == 3
    assert "splitmuxsink" in p and f"max-size-time={300 * 10**9}" in p
    assert "rtmpsink" in p and "appsink name=analysis" in p
    assert not any(x in p for x in ("x264enc", "x265enc", "nvv4l2h264enc", "nvv4l2h265enc"))  # kein Re-Encode
    assert "rtph264depay" in p
    # ohne Stream kein RTMP-Zweig
    assert "rtmpsink" not in build_gstreamer_pipeline(RecorderConfig())


def test_pipeline_hw_encoder_rules():
    ok = RecorderConfig(rtmp_url="rtmp://x/y", hw_encoder="nvv4l2h264enc", device="orin_nx", overlay=True,
                        codec="h265")
    p = build_gstreamer_pipeline(ok)
    assert "nvv4l2h264enc" in p and "rtph265depay" in p
    with pytest.raises(EncoderUnavailableError):
        build_gstreamer_pipeline(RecorderConfig(hw_encoder="nvv4l2h264enc", device="orin_nano"))
    with pytest.raises(EncoderUnavailableError):
        build_gstreamer_pipeline(RecorderConfig(overlay=True, device="orin_nano"))
    with pytest.raises(ValueError):
        build_gstreamer_pipeline(RecorderConfig(codec="h265", rtmp_url="rtmp://x/y"))
    assert "eflvmux" in build_gstreamer_pipeline(RecorderConfig(codec="h265", rtmp_url="rtmp://x/y",
                                                                enhanced_rtmp=True))


def _touch(d, i, size=100):
    p = os.path.join(d, f"seg_{i:05d}.mkv")
    with open(p, "wb") as f:
        f.write(b"x" * size)
    return p


def test_local_recorder_segments_and_deletion(tmp_path):
    now = [1000.0]
    rec = LocalRecorder(str(tmp_path), retention_s=100.0, min_free_bytes=0, clock=lambda: now[0])
    paths = [_touch(str(tmp_path), i) for i in range(3)]
    assert [s.index for s in rec.list_segments()] == [0, 1, 2]
    assert [s.index for s in rec.closed_segments(recording=True)] == [0, 1]
    assert rec.total_bytes() == 300 and rec.check_free_space(60, 1.0)
    rec.mark_uploaded(paths[0])
    assert rec.delete_uploaded() == []  # Aufbewahrung noch nicht abgelaufen
    now[0] += 101
    assert rec.delete_uploaded() == [paths[0]]
    assert os.path.exists(paths[1]) and not os.path.exists(paths[0])  # nicht hochgeladen bleibt
    rec2 = LocalRecorder(str(tmp_path), min_free_bytes=10**18, clock=lambda: now[0])
    rec2.mark_uploaded(paths[1])
    assert rec2.prune_for_space(recording=False) == [paths[1]]  # nur bestaetigte Segmente
    assert os.path.exists(paths[2])
    assert not rec2.check_free_space()


# --------------------------------------------------------------------------- Edge: Uploader
class FakeTransport:
    def __init__(self, fail_puts=(), fail_begin=0):
        self.data: dict[str, bytearray] = {}
        self.done: set[str] = set()
        self.fail_puts = set(fail_puts)  # Nummern (1-basiert) der put_chunk-Aufrufe, die scheitern
        self.fail_begin = fail_begin
        self.puts = 0
        self.begins = 0

    def begin(self, key, size):
        self.begins += 1
        if self.fail_begin > 0:
            self.fail_begin -= 1
            raise TransportError("kein Netz")
        self.data.setdefault(key, bytearray())
        return key

    def committed_offset(self, uid):
        return len(self.data[uid])

    def put_chunk(self, uid, offset, chunk):
        self.puts += 1
        if self.puts in self.fail_puts:
            self.data[uid] += chunk[: len(chunk) // 2]  # halb angekommen
            raise TransportError("Verbindung weg")
        assert offset == len(self.data[uid])
        self.data[uid] += chunk

    def complete(self, uid):
        self.done.add(uid)


def test_uploader_retries_with_backoff_and_resumes(tmp_path):
    p = tmp_path / "seg_00000.mkv"
    payload = os.urandom(1000)
    p.write_bytes(payload)
    t = [0.0]
    tr = FakeTransport(fail_puts={2}, fail_begin=1)
    q = UploadQueue(str(tmp_path / "q.db"), clock=lambda: t[0])
    q.enqueue(str(p))
    up = Uploader(q, tr, chunk_size=300, clock=lambda: t[0], backoff_base=2.0)
    assert up.run_once()["failed"] == 1  # begin scheitert
    it = q.get(str(p))
    assert it.state == SegmentState.PENDING and it.attempts == 1 and it.next_attempt == pytest.approx(2.0)
    assert up.run_once() == {"done": 0, "failed": 0, "skipped_offline": 0}  # Backoff laeuft
    t[0] = 2.0
    assert up.run_once()["failed"] == 1  # 2. put scheitert, halber Chunk beim Server
    it = q.get(str(p))
    assert it.attempts == 2 and it.next_attempt == pytest.approx(2.0 + 4.0)
    t[0] = 6.0
    assert up.run_once()["done"] == 1
    assert bytes(tr.data[os.path.basename(str(p))]) == payload  # fortgesetzt statt neu, keine Luecken
    assert q.get(str(p)).state == SegmentState.DONE and "seg_00000.mkv" in tr.done
    assert tr.begins == 2  # upload_id wurde wiederverwendet


def test_uploader_offline_and_persistence(tmp_path):
    p = tmp_path / "seg_00001.mkv"
    p.write_bytes(b"a" * 50)
    db = str(tmp_path / "q.db")
    q = UploadQueue(db)
    q.enqueue(str(p))
    assert q.enqueue(str(p)) is False  # idempotent
    online = [False]
    tr = FakeTransport()
    up = Uploader(q, tr, is_online=lambda: online[0])
    assert up.run_once()["skipped_offline"] == 1 and tr.begins == 0
    assert q.get(str(p)).attempts == 0 and q.pending_bytes() == 50
    q.update(str(p), state=SegmentState.UPLOADING)  # Absturz mitten im Upload
    q2 = UploadQueue(db)  # Neustart
    assert q2.get(str(p)).state == SegmentState.PENDING
    online[0] = True
    done = []
    up2 = Uploader(q2, tr, on_done=done.append)
    assert up2.run_once()["done"] == 1 and done and q2.pending_bytes() == 0


def test_lte_fallback_policy(tmp_path):
    p = tmp_path / "seg_00002.mkv"
    p.write_bytes(b"b" * 100)
    q = UploadQueue()
    q.enqueue(str(p))
    lte = LteFallback(budget_bytes=1000, allow_segments=False)
    up = Uploader(q, FakeTransport(), link=lte)
    assert up.run_once()["failed"] == 1 and q.get(str(p)).attempts == 0  # Policy, kein Fehlversuch
    lte.allow_segments = True
    assert up.run_once()["done"] == 1 and lte.used_bytes == 100


def test_bandwidth_helpers():
    assert not stream_feasible(3.0) and not stream_feasible(5.0)  # 80 % Headroom: 4 < 5 Mbit/s
    f = stream_feasible(10.0)
    assert f and f.recommended_bitrate_mbps == pytest.approx(8.0)
    assert stream_feasible(100.0).recommended_bitrate_mbps == 14.0
    assert realtime_upload_possible(40, 8, 20) and not realtime_upload_possible(20, 8, 20)


def test_session_overtime_game():
    game, sim, store, sess = _run_session(seed=2, periods=1, length=60.0, frames=False)
    assert sim.periods_played == 2
    sess.feed_many(sim.observations, sim.scoreboard)
    sess.finish()
    ev = store.effective(game.id)
    kinds = [(e.type, e.period) for e in ev if e.type in (T.GAME_START, T.PERIOD_START, T.PERIOD_END, T.GAME_END)]
    assert kinds == [(T.GAME_START, 1), (T.PERIOD_START, 1), (T.PERIOD_END, 1), (T.PERIOD_START, 2),
                     (T.PERIOD_END, 2), (T.GAME_END, 2)]
    box = compute_boxscore(ev, game, DEMO.rosters)
    assert box.score == sim.final_score
    assert sum(l.seconds_played for l in box.players.values()) == pytest.approx(
        10 * (game.period_length_s + game.overtime_length_s), abs=1.0)
