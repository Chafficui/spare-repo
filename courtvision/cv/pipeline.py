"""CVPipeline: Frames/Detektionen -> geordneter Strom von ``Observation``s.

Ablauf je Frame: Detektor -> Tracker -> Teams -> Trikotnummern/Kader ->
Kalibrierung (Feldkoordinaten) -> Ballbesitz / Wurf / Ring / Aus / Formation.
FrameState + Possession etc. entstehen mit ``state_hz`` (10 Hz); Wurf- und
Ring-Detektoren laufen bei jedem Aufruf (volle Bildrate).

Betriebsarten (``PipelineConfig``):

* **EDGE (live)**: Detektion auf herunterskaliertem Vollbild (``detect_max_side``
  1280), Nummern nur jeden 5. Frame je Track, keine Pose (Besitz ueber
  Bodenabstand), Ring-ROIs in voller Aufloesung. Ziel: Echtzeit auf Edge-GPU.
* **CLOUD (Nachlauf)**: Re-Run der aufgezeichneten Spiele in voller Aufloesung
  (``detect_max_side=None``), dichteres Nummernlesen (jeder 3. Frame), Pose fuer
  Hand-Keypoints. Liefert die korrigierten Daten nach Abpfiff.

``process_tracks`` nimmt bereits detektierte Frames (ohne Modelle) und ist der
Testeinstieg. Ausgaben laufen durch einen Reorder-Puffer (``reorder_delay`` s),
damit rueckdatierte Ereignisse (z. B. Besitzwechsel mit Zeitstempel des ersten
Frames der Serie) zeitlich sortiert herauskommen; ``flush()`` leert ihn.

Datenschutz: Re-ID-Embeddings und Trikot-Stimmen leben nur im Speicher des
laufenden Spiels; ``end_game()`` verwirft sie. Keine Gesichtserkennung.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np

from ..domain.models import Game, Player, Side, Team
from ..domain.observations import (Basket, ClockState, FrameState, Observation, PlayerState, obs_ts)
from ..geometry.court import Calibration, in_court
from .ball import BallSample, RimMakeDetector, ShotDetector
from .detection import Detection, Detector, downscale
from .jersey import (Assignment, JerseyNumberReader, JerseyReading, NumberVoter, RosterAssigner, RosterEntry,
                     reading_from_probs)
from .pose import PoseEstimator, foot_point, hand_points
from .possession import FormationDetector, OutOfBoundsDetector, PossessionEstimator
from .teams import RGB, TeamAssigner, color_histogram_embedding, parse_color
from .tracking import Tracker, TrackerConfig


class Mode(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


@dataclass
class RimConfig:
    """Permanente Ring-ROI (Volbild-Pixel): Mittelpunkt und Ringradius in Pixeln."""

    center_uv: tuple[float, float]
    radius_px: float
    roi_scale: float = 3.0

    def contains(self, uv: tuple[float, float]) -> bool:
        r = self.radius_px * self.roi_scale
        return abs(uv[0] - self.center_uv[0]) <= r and abs(uv[1] - self.center_uv[1]) <= r


@dataclass
class PipelineConfig:
    mode: Mode = Mode.EDGE
    state_hz: float = 10.0
    detect_max_side: Optional[int] = 1280
    jersey_sample_every: int = 5
    use_pose: bool = False
    court_margin: float = 2.0  # Tracks weiter ausserhalb (Bank/Zuschauer) werden ignoriert
    team_refit_every: int = 30
    roster_every: int = 10
    ball_min_score: float = 0.3
    reorder_delay: float = 0.5
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    @classmethod
    def edge(cls, **kw) -> "PipelineConfig":
        return cls(mode=Mode.EDGE, **kw)

    @classmethod
    def cloud(cls, **kw) -> "PipelineConfig":
        base = dict(mode=Mode.CLOUD, detect_max_side=None, jersey_sample_every=3, use_pose=True)
        base.update(kw)
        return cls(**base)  # type: ignore[arg-type]


ReadingLike = Union[JerseyReading, tuple, np.ndarray]


class CVPipeline:
    def __init__(self, calibration: Calibration, config: Optional[PipelineConfig] = None, *,
                 detector: Optional[Detector] = None, pose: Optional[PoseEstimator] = None,
                 jersey_reader: Optional[JerseyNumberReader] = None,
                 embedder: Callable[[np.ndarray], np.ndarray] = color_histogram_embedding,
                 seed_colors: Optional[Mapping[Side, RGB]] = None,
                 team_assigner: Optional[TeamAssigner] = None,
                 rosters: Optional[Mapping[Side, Sequence[RosterEntry]]] = None,
                 rims: Optional[Mapping[Basket, RimConfig]] = None) -> None:
        self.cal = calibration
        self.cfg = config or PipelineConfig()
        self.detector, self.pose, self.reader, self.embedder = detector, pose, jersey_reader, embedder
        self.tracker = Tracker(self.cfg.tracker)
        self.teams = team_assigner or TeamAssigner(seed_colors=seed_colors)
        self.voter = NumberVoter(sample_every=self.cfg.jersey_sample_every)
        self.roster = RosterAssigner(rosters) if rosters else None
        self.possession = PossessionEstimator()
        self.oob = OutOfBoundsDetector()
        self.formation = FormationDetector()
        self.shots = ShotDetector()
        self.rims = dict(rims or {})
        self.rim_detectors = {b: RimMakeDetector(b, r.center_uv, r.radius_px) for b, r in self.rims.items()}
        self.identities: dict[int, Assignment] = {}
        self.clock: Optional[ClockState] = None
        self.frame_no = 0
        self._last_tick = -1e9
        self._heap: list[tuple[float, int, Observation]] = []
        self._seq = 0

    @classmethod
    def from_game(cls, calibration: Calibration, game: Game, teams: Mapping[Side, Team], players: Sequence[Player],
                  config: Optional[PipelineConfig] = None, unavailable: Iterable[str] = (), **kw) -> "CVPipeline":
        """Spielanlage -> Seed-Farben (Team.jersey_color) und Kader je Seite."""
        seeds = {s: parse_color(t.jersey_color) for s, t in teams.items() if t.jersey_color}
        out = set(unavailable)
        rosters = {s: [RosterEntry.from_player(p, p.id not in out) for p in players if p.team_id == game.team_id(s)]
                   for s in teams}
        return cls(calibration, config, seed_colors=seeds if len(seeds) == 2 else None, rosters=rosters, **kw)

    # -- Zustand von aussen -----------------------------------------------------
    def set_clock(self, clock: ClockState) -> None:
        self.clock = clock

    @property
    def clock_stopped(self) -> bool:
        """Ohne ClockState (Anzeigetafel) ist unbekannt -> False (kein Aus, kein Wechselfenster)."""
        return self.clock is not None and not self.clock.running

    def end_game(self) -> list[Observation]:
        """Spielende: Puffer leeren und alle transienten Merkmale verwerfen."""
        out = self.flush()
        self.tracker.clear_embeddings()
        self.teams.clear()
        self.voter.clear()
        if self.roster:
            self.roster.clear()
        self.identities.clear()
        return out

    # -- Eingang 1: Rohframes -----------------------------------------------------
    def process_frame(self, frame: np.ndarray, ts: float) -> list[Observation]:
        if self.detector is None:
            raise RuntimeError("process_frame braucht einen Detector; ohne Modelle process_tracks() verwenden")
        ds = downscale(frame, max_side=self.cfg.detect_max_side)
        dets = [d.scaled(ds.scale_x, ds.scale_y) for d in self.detector.detect(ds.image)]
        h, w = frame.shape[:2]
        embs: list[Optional[np.ndarray]] = []
        for d in dets:
            if d.cls in ("player", "referee"):
                x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
                crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                embs.append(self.embedder(crop) if crop.size else None)
            else:
                embs.append(None)
        poses = None
        if self.cfg.use_pose and self.pose is not None:
            idx = [i for i, d in enumerate(dets) if d.cls == "player"]
            kps = self.pose.estimate(frame, [dets[i].bbox for i in idx])
            poses = dict(zip(idx, kps))
        return self._run(ts, dets, embs, None, poses, frame)

    def process_video(self, frames: Iterable[tuple[float, np.ndarray]]) -> Iterator[Observation]:
        for ts, fr in frames:
            yield from self.process_frame(fr, ts)
        yield from self.flush()

    # -- Eingang 2: fertige Detektionen (testbar ohne Modelle) -------------------------
    def process_tracks(self, ts: float, detections: Sequence[Detection],
                       embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
                       jersey_readings: Optional[Mapping[int, ReadingLike]] = None,
                       poses: Optional[Mapping[int, np.ndarray]] = None) -> list[Observation]:
        """Detektionen in Volbild-Pixeln. ``embeddings`` parallel zu ``detections``;
        ``jersey_readings``/``poses`` nach Detektionsindex."""
        return self._run(ts, list(detections), embeddings, jersey_readings, poses, None)

    # -- Kern --------------------------------------------------------------------------
    def _run(self, ts, dets, embeddings, readings, poses, frame) -> list[Observation]:
        cfg = self.cfg
        self.frame_no += 1
        trk_idx = [i for i, d in enumerate(dets) if d.cls in ("player", "referee")]
        trk_dets = [dets[i] for i in trk_idx]
        trk_embs = None if embeddings is None else [embeddings[i] for i in trk_idx]
        tracks = self.tracker.update(trk_dets, trk_embs)
        det_of: dict[int, int] = {}  # Track-ID -> Originalindex der Detektion
        for t in tracks:
            if t.det_index is not None:
                det_of[t.id] = trk_idx[t.det_index]

        # Court-Position; Bank/Zuschauer ausfiltern
        on_court: dict[int, tuple[float, float]] = {}
        for t in tracks:
            fx, fy = t.foot_point
            if poses and t.id in det_of and det_of[t.id] in poses:
                fp = foot_point(poses[det_of[t.id]])
                if fp is not None:
                    fx, fy = fp
            x, y = self.cal.to_court(fx, fy)
            if in_court(x, y, cfg.court_margin):
                on_court[t.id] = (x, y)
        live = [t for t in tracks if t.id in on_court]

        # Teams
        for t in live:
            if t.embedding is not None:
                self.teams.observe(t.id, t.embedding, is_referee=(t.cls == "referee"))
        if (not self.teams.fitted and self.frame_no % 5 == 0) or self.frame_no % cfg.team_refit_every == 0:
            self.teams.fit()
        team_of = {t.id: (None if t.cls == "referee" else self.teams.team_of(t.id)) for t in live}

        # Trikotnummern
        if readings:
            for tid, di in det_of.items():
                if di in readings:
                    self._add_reading(tid, readings[di])
        if frame is not None and self.reader is not None:
            for t in live:
                if t.cls == "player" and self.voter.should_sample(t.id, self.frame_no):
                    x1, y1, x2, y2 = t.bbox
                    bh = y2 - y1
                    ya, yb = int(max(y1 + 0.12 * bh, 0)), int(max(y1 + 0.6 * bh, 0))
                    xa, xb = int(max(x1, 0)), int(max(x2, 0))
                    crop = frame[ya:yb, xa:xb]
                    if crop.size:
                        self._add_reading(t.id, self.reader.read(crop))
        if self.roster and self.frame_no % cfg.roster_every == 0:
            for side in ("home", "away"):
                ids = [t.id for t in live if team_of[t.id] == side]
                res = self.roster.assign(side, {i: self.voter.votes(i) for i in ids}, self.clock_stopped, ts)  # type: ignore[arg-type]
                self.identities.update(res)

        players = tuple(
            PlayerState(t.id, team_of[t.id], on_court[t.id][0], on_court[t.id][1],
                        self.identities[t.id].player_id if t.id in self.identities else None,
                        self.identities[t.id].confidence if t.id in self.identities else 0.0)
            for t in sorted(live, key=lambda t: t.id))

        # Ball
        balls = [d for d in dets if d.cls == "ball" and d.score >= cfg.ball_min_score]
        ball = max(balls, key=lambda d: d.score) if balls else None
        ball_uv = ball.center if ball else None
        ball_plain = self.cal.to_court(*ball_uv) if ball_uv else None
        ball_hold = self._ball_ground(ball_uv, live) if ball_uv else None

        # Ring-Detektoren: volle Rate
        out: list[Observation] = []
        for b, det in self.rim_detectors.items():
            uv = ball_uv if (ball_uv and self.rims[b].contains(ball_uv)) else None
            m = det.update(ts, uv)
            if m:
                out.append(m)

        # 10-Hz-Takt
        poss_events = []
        if ts - self._last_tick >= 1.0 / cfg.state_hz - 1e-6:
            self._last_tick = ts
            hands = None
            if poses:
                hands = {}
                for tid, di in det_of.items():
                    if di in poses and tid in on_court:
                        hp = hand_points(poses[di])
                        if hp:
                            hands[tid] = [self.cal.to_court(*p) for p in hp]
            pe = self.possession.update(ts, players, ball_hold, hands)
            if pe:
                poss_events.append(pe)
                out.append(pe)
            oob = self.oob.update(ts, ball_plain, play_stopped=self.clock_stopped,
                                  last_touch=self.possession.last_touch)
            if oob:
                out.append(oob)
            fm = self.formation.update(ts, players)
            if fm:
                out.append(fm)
            out.append(FrameState(ts, players, ball_plain[0] if ball_plain else None,
                                  ball_plain[1] if ball_plain else None, None))

        sample = BallSample(ts, ball_plain[0], ball_plain[1], None, ball_uv[1]) if ball_uv and ball_plain else None
        out.extend(self.shots.update(ts, sample, players, poss_events))

        for o in out:
            self._seq += 1
            heapq.heappush(self._heap, (obs_ts(o), self._seq, o))
        return self._pop(ts - cfg.reorder_delay)

    def _add_reading(self, track_id: int, r: ReadingLike) -> None:
        if isinstance(r, np.ndarray):
            r = reading_from_probs(r)
        elif isinstance(r, tuple):
            r = JerseyReading(*r)
        self.voter.add_reading(track_id, r.number, r.confidence)

    def _ball_ground(self, uv, tracks) -> Optional[tuple[float, float]]:
        """Bodenpunkt des Balls fuer die Besitzpruefung: liegt der Ball im (erweiterten)
        Bild-Rechteck eines Spielers, wird er senkrecht auf dessen Fusslinie
        projiziert (Hand ~1-2 m ueber dem Boden verschiebt sonst die Projektion)."""
        best, bd = None, 1e18
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            w, h = x2 - x1, y2 - y1
            if x1 - 0.3 * w <= uv[0] <= x2 + 0.3 * w and y1 - 0.15 * h <= uv[1] <= y2 + 0.05 * h:
                d = abs(uv[0] - (x1 + x2) / 2)
                if d < bd:
                    best, bd = t, d
        if best is None:
            return self.cal.to_court(*uv)
        return self.cal.to_court(uv[0], best.foot_point[1])

    def _pop(self, upto: float) -> list[Observation]:
        out = []
        while self._heap and self._heap[0][0] <= upto:
            out.append(heapq.heappop(self._heap)[2])
        return out

    def flush(self) -> list[Observation]:
        return self._pop(float("inf"))
