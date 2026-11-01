"""Multi-Objekt-Tracker im Stil von OC-SORT / BoT-SORT (reines numpy/scipy).

* Kalman-Filter mit konstanter Geschwindigkeit je Track (Zustand cx, cy, w, h + Raten)
* Kosten: 1-IoU (optional gemischt mit Embedding-Kosinus und OC-SORT-Richtungsterm)
* Hungarian-Zuordnung (scipy), zweite Stufe gegen die *letzte Beobachtung*
  verlorener Tracks (OC-SORT: beobachtungszentriert)
* Observation-centric Re-Update (ORU): nach einer Verdeckung wird der Filter auf
  den Zustand der letzten Beobachtung zurueckgesetzt und mit einer linear
  interpolierten virtuellen Trajektorie neu aktualisiert -> kein Drift.
* Geburt nach ``min_hits``, Loeschung nach ``max_age`` Frames ohne Beobachtung.

Datenschutz: Embeddings (Re-ID) liegen nur transient im Speicher des laufenden
Spiels; ``Tracker.clear_embeddings()`` am Spielende, ``Track`` pickelt sie nie.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import Detection

_BIG = 1e6


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _to_z(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], float)


def _to_box(x: np.ndarray) -> np.ndarray:
    w, h = max(x[2], 1.0), max(x[3], 1.0)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxFilter:
    """Konstante Geschwindigkeit auf (cx, cy, w, h); Rauschen skaliert mit der Boxhoehe."""

    def __init__(self, bbox, std_pos: float = 0.05, std_vel: float = 0.02, std_meas: float = 0.03) -> None:
        self.std_pos, self.std_vel, self.std_meas = std_pos, std_vel, std_meas
        z = _to_z(np.asarray(bbox, float))
        self.x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        self.P = np.diag(np.concatenate([(2 * std_meas * h) ** 2 * np.ones(4), (10 * std_vel * h) ** 2 * np.ones(4)]))
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def predict(self) -> None:
        h = max(self.x[3], 1.0)
        Q = np.diag(np.concatenate([(self.std_pos * h) ** 2 * np.ones(4), (self.std_vel * h) ** 2 * np.ones(4)]))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, z: np.ndarray) -> None:
        h = max(self.x[3], 1.0)
        R = np.diag((self.std_meas * h) ** 2 * np.ones(4))
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def snapshot(self):
        return self.x.copy(), self.P.copy()

    def restore(self, snap) -> None:
        self.x, self.P = snap[0].copy(), snap[1].copy()


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = np.linalg.norm(v)
    return None if n < 1e-6 else v / n


@dataclass
class Track:
    kf: KalmanBoxFilter = field(repr=False)
    cls: str = "player"
    score: float = 1.0
    id: int = -1  # wird erst bei Bestaetigung vergeben
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    det_index: Optional[int] = None  # Index der zugeordneten Detektion im aktuellen Frame
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    last_obs: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    obs_dir: Optional[np.ndarray] = field(default=None, repr=False)
    _saved: tuple = field(default=None, repr=False)  # type: ignore[assignment]
    _obs_hist: list = field(default_factory=list, repr=False)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        b = _to_box(self.kf.x)
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    @property
    def foot_point(self) -> tuple[float, float]:
        b = self.bbox
        return ((b[0] + b[2]) / 2, b[3])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.kf.x[4]), float(self.kf.x[5])

    def __getstate__(self):
        """Persistenz-Schutz: Re-ID-Embeddings werden nie mitgeserialisiert."""
        s = self.__dict__.copy()
        s["embedding"] = None
        return s


@dataclass
class TrackerConfig:
    min_hits: int = 3
    max_age: int = 30
    tentative_max_age: int = 2
    iou_min: float = 0.1
    max_cost: float = 0.9
    embedding_weight: float = 0.3
    embedding_ema: float = 0.9
    ocm_weight: float = 0.2  # Richtungskonsistenz (OC-SORT OCM)
    oru: bool = True  # observation-centric re-update
    last_obs_stage: bool = True  # 2. Stufe auf letzter Beobachtung


class Tracker:
    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.cfg = config or TrackerConfig()
        self.tracks: list[Track] = []
        self.frame = 0
        self._next_id = 1

    # -- Hilfen ---------------------------------------------------------
    @staticmethod
    def _emb_cost(tracks: Sequence[Track], embs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if embs is None:
            return None
        C = np.full((len(tracks), len(embs)), 0.5)
        for i, t in enumerate(tracks):
            if t.embedding is None:
                continue
            for j, e in enumerate(embs):
                if e is None:
                    continue
                C[i, j] = 1.0 - float(np.dot(t.embedding, e))
        return np.clip(C, 0.0, 1.0)  # Kosinus-Distanz; gleiche Person ~0, andere ~1

    def _cost(self, boxes_t: np.ndarray, tracks: Sequence[Track], dets: np.ndarray,
              embs, det_centers: np.ndarray, use_ocm: bool) -> np.ndarray:
        iou = iou_matrix(boxes_t, dets)
        cost = 1.0 - iou
        ec = self._emb_cost(tracks, embs)
        if ec is not None:
            w = self.cfg.embedding_weight
            cost = (1 - w) * cost + w * np.minimum(ec, 1.0)
        if use_ocm and self.cfg.ocm_weight > 0:
            for i, t in enumerate(tracks):
                if t.obs_dir is None or t.last_obs is None:
                    continue
                lc = _to_z(t.last_obs)[:2]
                for j in range(len(dets)):
                    d = _unit(det_centers[j] - lc)
                    if d is None:
                        continue
                    ang = np.arccos(np.clip(np.dot(t.obs_dir, d), -1, 1)) / np.pi
                    cost[i, j] += self.cfg.ocm_weight * ang
        cost[iou < self.cfg.iou_min] = _BIG
        return cost

    def _solve(self, cost: np.ndarray):
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        r, c = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(r, c) if cost[i, j] <= self.cfg.max_cost]
        mi, mj = {i for i, _ in matches}, {j for _, j in matches}
        return matches, [i for i in range(cost.shape[0]) if i not in mi], [j for j in range(cost.shape[1]) if j not in mj]

    def _apply_update(self, t: Track, box: np.ndarray, det: Detection, j: int, emb) -> None:
        cfg = self.cfg
        if cfg.oru and t.time_since_update > 1 and t._saved is not None and t.last_obs is not None:
            gap = t.time_since_update
            t.kf.restore(t._saved)
            for k in range(1, gap):
                a = k / gap
                t.kf.predict()
                t.kf.update(_to_z((1 - a) * t.last_obs + a * box))
            t.kf.predict()
        t.kf.update(_to_z(box))
        # Beobachtungsrichtung (OCM): ueber ein paar Frames hinweg gemittelt
        c = _to_z(box)[:2]
        t._obs_hist.append(c)
        t._obs_hist[:] = t._obs_hist[-4:]
        if len(t._obs_hist) >= 2:
            t.obs_dir = _unit(t._obs_hist[-1] - t._obs_hist[0])
        t.last_obs = box.copy()
        t._saved = t.kf.snapshot()
        t.time_since_update = 0
        t.hits += 1
        t.score, t.cls, t.det_index = det.score, det.cls, j
        if emb is not None:
            e = np.asarray(emb, float)
            e = e / (np.linalg.norm(e) + 1e-12)
            t.embedding = e if t.embedding is None else _renorm(cfg.embedding_ema * t.embedding + (1 - cfg.embedding_ema) * e)
        if not t.confirmed and t.hits >= cfg.min_hits:
            t.confirmed, t.id = True, self._next_id
            self._next_id += 1

    # -- Hauptschritt ---------------------------------------------------
    def update(self, detections: Sequence[Detection], embeddings: Optional[Sequence[Optional[np.ndarray]]] = None) -> list[Track]:
        """Ein Frame. Rueckgabe: bestaetigte Tracks mit Beobachtung in diesem Frame."""
        cfg = self.cfg
        self.frame += 1
        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1
            t.det_index = None
        dets = np.array([d.bbox for d in detections], float).reshape(-1, 4)
        centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in dets]).reshape(-1, 2)
        embs = None
        if embeddings is not None:
            embs = np.empty(len(detections), dtype=object)
            for j, e in enumerate(embeddings):
                embs[j] = None if e is None else _renorm(np.asarray(e, float))
            if all(e is None for e in embs):
                embs = None

        matches: list[tuple[int, int]] = []
        un_t = list(range(len(self.tracks)))
        un_d = list(range(len(detections)))
        if self.tracks and len(detections):
            pred = np.array([t.bbox for t in self.tracks])
            sub_e = embs
            C = self._cost(pred, self.tracks, dets, sub_e, centers, use_ocm=True)
            matches, un_t, un_d = self._solve(C)
        # Stufe 2: verlorene Tracks gegen letzte Beobachtung
        if cfg.last_obs_stage and un_t and un_d:
            lost = [i for i in un_t if self.tracks[i].last_obs is not None]
            if lost:
                lo = np.array([self.tracks[i].last_obs for i in lost])
                sub_d = np.array([dets[j] for j in un_d])
                sub_e = None if embs is None else embs[un_d]
                C = self._cost(lo, [self.tracks[i] for i in lost], sub_d, sub_e, centers[un_d], use_ocm=False)
                m2, _, _ = self._solve(C)
                used_t, used_d = set(), set()
                for a, b in m2:
                    matches.append((lost[a], un_d[b]))
                    used_t.add(lost[a]); used_d.add(un_d[b])
                un_t = [i for i in un_t if i not in used_t]
                un_d = [j for j in un_d if j not in used_d]
        for i, j in matches:
            self._apply_update(self.tracks[i], dets[j], detections[j], j, None if embs is None else embs[j])
        for j in un_d:
            t = Track(KalmanBoxFilter(dets[j]), cls=detections[j].cls, score=detections[j].score, det_index=j)
            t.last_obs, t._saved = dets[j].copy(), t.kf.snapshot()
            t._obs_hist = [_to_z(dets[j])[:2]]
            if embs is not None and embs[j] is not None:
                t.embedding = embs[j]
            if cfg.min_hits <= 1:
                t.confirmed, t.id = True, self._next_id
                self._next_id += 1
            self.tracks.append(t)
        self.tracks = [t for t in self.tracks if not (
            t.time_since_update > (cfg.max_age if t.confirmed else cfg.tentative_max_age))]
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def clear_embeddings(self) -> None:
        """Spielende: alle Re-ID-Merkmale verwerfen (Datenschutz)."""
        for t in self.tracks:
            t.embedding = None


def _renorm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)
