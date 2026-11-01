"""Trikotnummern: Leser (Protocol), Abstimmung je Track, Zuordnung zum Kader.

Der eigentliche Nummern-Klassifikator (Klassen 0-99 + "unlesbar") ist ein
trainiertes Modell und hier nur als Protocol/optionaler Torch-Adapter definiert.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..domain.models import Player, Side

UNREADABLE = 100  # Klassenindex "unlesbar" (Klassen 0..99 sind Nummern)


@dataclass(frozen=True)
class JerseyReading:
    number: Optional[int]  # None = unlesbar
    confidence: float = 1.0


@runtime_checkable
class JerseyNumberReader(Protocol):
    def read(self, crop: np.ndarray) -> np.ndarray:
        """Klassenwahrscheinlichkeiten, Form (101,): Index 0..99 = Nummer, 100 = unlesbar."""
        ...


def reading_from_probs(probs: np.ndarray, min_conf: float = 0.0) -> JerseyReading:
    p = np.asarray(probs, float)
    k = int(p.argmax())
    if k == UNREADABLE or p[k] < min_conf:
        return JerseyReading(None, float(p[UNREADABLE]) if len(p) > UNREADABLE else 0.0)
    return JerseyReading(k, float(p[k]))


class TorchJerseyReader:  # pragma: no cover - benoetigt torch + Modell
    """Optionaler Adapter: TorchScript-Klassifikator (Eingabe 3xHxW, 101 Klassen)."""

    def __init__(self, checkpoint_path: str, input_size: tuple[int, int] = (64, 64), device: str = "cpu") -> None:
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError("TorchJerseyReader benoetigt 'torch'.") from e
        self._torch, self.size, self.device = torch, input_size, device
        self.model = torch.jit.load(checkpoint_path, map_location=device).eval()

    def read(self, crop: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = self.size
        ys = np.minimum((np.arange(h) * crop.shape[0] / h).astype(int), crop.shape[0] - 1)
        xs = np.minimum((np.arange(w) * crop.shape[1] / w).astype(int), crop.shape[1] - 1)
        x = torch.from_numpy(crop[ys][:, xs].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=-1)[0].cpu().numpy()


@dataclass
class TrackVotes:
    """Stimmenverteilung eines Tracks: Gewicht (Summe Konfidenz) je Nummer."""

    numbers: dict[int, float] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    unreadable: float = 0.0
    n_readings: int = 0
    confirmed: Optional[int] = None

    @property
    def number_mass(self) -> float:
        return float(sum(self.numbers.values()))


class NumberVoter:
    """Jeden ``sample_every``-ten Frame je Track lesen; Nummer gilt nach
    ``confirm_count`` identischen Lesungen (mit Konfidenz >= ``min_conf``)."""

    def __init__(self, sample_every: int = 5, confirm_count: int = 3, min_conf: float = 0.5) -> None:
        self.sample_every, self.confirm_count, self.min_conf = sample_every, confirm_count, min_conf
        self._votes: dict[int, TrackVotes] = {}
        self._last_sample: dict[int, int] = {}

    def should_sample(self, track_id: int, frame_idx: int) -> bool:
        last = self._last_sample.get(track_id)
        if last is None or frame_idx - last >= self.sample_every:
            self._last_sample[track_id] = frame_idx
            return True
        return False

    def add_reading(self, track_id: int, number: Optional[int], confidence: float = 1.0) -> Optional[int]:
        """Lesung eintragen; gibt die (ggf. neu) bestaetigte Nummer zurueck."""
        v = self._votes.setdefault(track_id, TrackVotes())
        v.n_readings += 1
        if number is None:
            v.unreadable += confidence
        else:
            if not 0 <= number <= 99:
                raise ValueError("Nummer ausserhalb 0..99")
            v.numbers[number] = v.numbers.get(number, 0.0) + confidence
            if confidence >= self.min_conf:
                v.counts[number] = v.counts.get(number, 0) + 1
        ok = [n for n, c in v.counts.items() if c >= self.confirm_count]
        v.confirmed = max(ok, key=lambda n: (v.counts[n], v.numbers[n])) if ok else None
        return v.confirmed

    def add_probs(self, track_id: int, probs: np.ndarray) -> Optional[int]:
        r = reading_from_probs(probs)
        return self.add_reading(track_id, r.number, r.confidence)

    def confirmed(self, track_id: int) -> Optional[int]:
        v = self._votes.get(track_id)
        return v.confirmed if v else None

    def votes(self, track_id: int) -> TrackVotes:
        return self._votes.get(track_id, TrackVotes())

    def drop(self, track_id: int) -> None:
        self._votes.pop(track_id, None)
        self._last_sample.pop(track_id, None)

    def clear(self) -> None:
        self._votes.clear(); self._last_sample.clear()


@dataclass(frozen=True)
class RosterEntry:
    player_id: str
    number: int
    available: bool = True  # False: nicht im aktiven Kader / disqualifiziert

    @classmethod
    def from_player(cls, p: Player, available: bool = True) -> "RosterEntry":
        return cls(p.id, p.number, available)


@dataclass
class Assignment:
    player_id: Optional[str]
    confidence: float
    candidates: list[tuple[str, float]] = field(default_factory=list)


class RosterAssigner:
    """Track <-> Kaderspieler je Team als Zuordnungsproblem (Hungarian).

    Kosten = -log P(Nummer | Stimmen). Randbedingungen:
    * ein Spieler hoechstens auf einem Track; nur verfuegbare Spieler
    * hoechstens ``max_on_court`` (5) verschiedene Spieler je Team auf dem Feld
    * Identitaetswechsel eines bereits zugeordneten Tracks und Verdraengen
      nicht sichtbarer Feldspieler nur bei stehender Uhr (Wechselfenster) oder
      bei sehr hoher Konfidenz (``high_conf``)
    * Tracks ohne ausreichende Lesung bekommen keine Identitaet; Kandidaten =
      Kader minus die aktuell zugeordneten Spieler anderer Tracks.
    """

    def __init__(self, rosters: Mapping[Side, Sequence[RosterEntry]], max_on_court: int = 5,
                 high_conf: float = 0.95, min_mass: float = 1.5, smoothing: float = 0.05) -> None:
        self.rosters = {s: list(r) for s, r in rosters.items()}
        self.max_on_court, self.high_conf = max_on_court, high_conf
        self.min_mass, self.smoothing = min_mass, smoothing
        self.track_player: dict[Side, dict[int, str]] = {s: {} for s in self.rosters}
        self.on_court: dict[Side, dict[str, float]] = {s: {} for s in self.rosters}  # player -> last_seen ts

    def clear(self) -> None:
        for s in self.rosters:
            self.track_player[s].clear(); self.on_court[s].clear()

    def _probs(self, v: TrackVotes, nums: Sequence[int]) -> np.ndarray:
        a = self.smoothing
        w = np.array([v.numbers.get(n, 0.0) for n in nums]) + a
        total = v.number_mass + a * len(nums)
        return w / total

    def assign(self, team: Side, votes: Mapping[int, TrackVotes], clock_stopped: bool,
               ts: float = 0.0) -> dict[int, Assignment]:
        roster = [r for r in self.rosters.get(team, []) if r.available]
        by_id = {r.player_id: r for r in roster}
        cur = self.track_player[team]
        court = self.on_court[team]
        # Tracks, die verschwunden sind, behalten ihre Zuordnung nicht; der Spieler bleibt auf dem Feld
        for tid in [t for t in cur if t not in votes]:
            del cur[tid]
        for pid in [p for p in court if p not in by_id]:  # inzwischen nicht verfuegbar
            del court[pid]
        for tid in [t for t, p in cur.items() if p not in by_id]:
            del cur[tid]
        nums = [r.number for r in roster]
        result: dict[int, Assignment] = {}

        readable = [t for t, v in votes.items() if v.confirmed is not None or v.number_mass >= self.min_mass]
        readable_set = set(readable)
        # sticky Tracks: Zuordnung bleibt solange Uhr laeuft und keine sehr sichere Gegenlesung
        free_cols = roster
        P = {t: self._probs(votes[t], nums) for t in readable}
        n_t, n_p = len(readable), len(free_cols)
        assigned: dict[int, str] = {}
        if n_t and n_p:
            cost = np.zeros((n_t, n_p))
            for i, t in enumerate(readable):
                for j, r in enumerate(free_cols):
                    c = -math.log(P[t][j])
                    if votes[t].numbers.get(r.number, 0.0) <= 0.0:
                        c = 1e6  # Nummer nie gelesen -> keine Zuordnung erzwingen
                    cp = cur.get(t)
                    if cp is not None and cp != r.player_id:
                        if not (clock_stopped or P[t][j] >= self.high_conf):
                            c = 1e6  # Identitaetswechsel nur im Wechselfenster
                    cost[i, j] = c
            # unlesbare Tracks mit alter Identitaet blockieren deren Spieler
            for t, pid in cur.items():
                if t not in readable_set and t in votes:
                    for j, r in enumerate(free_cols):
                        if r.player_id == pid:
                            cost[:, j] = 1e6
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e5:
                    assigned[readable[i]] = free_cols[j].player_id
        # Sticky fuer lesbare Tracks ohne Zuordnung (z. B. weil Wechsel verboten): alte Identitaet behalten
        for t in readable:
            if t not in assigned and t in cur:
                assigned[t] = cur[t]
        # unlesbare Tracks behalten ihre bisherige Identitaet
        for t, pid in cur.items():
            if t in votes and t not in readable_set:
                assigned[t] = pid
        # Max. 5 auf dem Feld: sichtbare + nicht sichtbare, noch nicht verdraengte Spieler
        visible_players = set(assigned.values())
        hidden = [p for p in court if p not in visible_players]
        new_entrants = [t for t in assigned if assigned[t] not in court]
        total = len(visible_players) + len(hidden)
        if total > self.max_on_court:
            conf_of = lambda t: P[t][nums.index(by_id[assigned[t]].number)] if t in P else 0.0  # noqa: E731
            very_sure = any(conf_of(t) >= self.high_conf for t in new_entrants)
            if clock_stopped or very_sure:
                hidden.sort(key=lambda p: court[p])  # am laengsten nicht gesehen zuerst
                while total > self.max_on_court and hidden:
                    del court[hidden.pop(0)]
                    total -= 1
            while total > self.max_on_court and new_entrants:
                weakest = min(new_entrants, key=lambda t: conf_of(t))
                new_entrants.remove(weakest)
                del assigned[weakest]
                total -= 1
        # Zustand + Ergebnis
        cur.clear()
        for t, pid in assigned.items():
            cur[t] = pid
            court[pid] = ts
        on_other = lambda tid: {p for t, p in assigned.items() if t != tid}  # noqa: E731
        for t, v in votes.items():
            pid = assigned.get(t)
            if pid is not None:
                conf = float(P[t][nums.index(by_id[pid].number)]) if t in P else 0.5
                order = sorted(zip(roster, P[t]), key=lambda x: -x[1]) if t in P else []
                cands = [(r.player_id, float(p)) for r, p in order[:3]]
                result[t] = Assignment(pid, conf, cands)
            else:
                taken = on_other(t)
                cands = [(r.player_id, 0.0) for r in roster if r.player_id not in taken]
                result[t] = Assignment(None, 0.0, cands)
        return result
