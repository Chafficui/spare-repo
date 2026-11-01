"""Deterministischer Spielgenerator: Ground-Truth-Events + CV-Beobachtungsstrom.

``simulate_game(seed, game, rosters)`` simuliert ein komplettes Basketballspiel
(Possessions, Wuerfe, Freiwuerfe, Rebounds, Fouls, Wechsel, Timeouts ...) und
liefert

* ``truth``: Events nach dem Schema in ``domain.events`` (CONFIRMED/CONSOLE),
* ``observations``: der Strom, den die CV-Pipeline erzeugen wuerde,
* ``scoreboard``: Anzeigetafel-Ablesungen (1 Hz) mit Verzoegerung beim Score.

Es gibt drei unabhaengige Zufallsquellen (Spiel, Rauschen, Frames). Daher
aendern ``noise``, ``include_frames`` und ``frame_hz`` die ``truth`` nie.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..domain.events import Event, EventSource, EventStatus, EventType, FoulKind, TurnoverKind
from ..domain.models import Game, Player, Side, SIDES, other_side
from ..domain.observations import (
    BallTouch, ClockState, Formation, FrameState, Observation, OutOfBounds, PlayerState,
    Possession, RimMake, ShotRelease,
)
from ..geometry import court as ct

SIDE_IDX = {"home": 0, "away": 1}
INF = float("inf")


def default_zones() -> dict[str, list[tuple[float, float]]]:
    """Bank-/Wechselzonen in Feldkoordinaten (Bank an der Seite y<0, Kampfgericht in der Mitte)."""
    return {
        "bench_home": [(-10.0, -10.5), (-1.8, -10.5), (-1.8, -7.7), (-10.0, -7.7)],
        "bench_away": [(1.8, -10.5), (10.0, -10.5), (10.0, -7.7), (1.8, -7.7)],
        "substitution": [(-1.5, -8.8), (1.5, -8.8), (1.5, -7.6), (-1.5, -7.6)],
    }


#: Maximale Haltezeit des Empfängers bis zum Wurf für einen Assist (s).
ASSIST_MAX_HOLD_S = 2.0


@dataclass
class SimNoise:
    """Degradationen des Beobachtungsstroms (Default: kein Rauschen)."""

    drop_player_id_prob: float = 0.0  # player_id=None in Beobachtungen
    miss_rim_make_prob: float = 0.0  # RimMake eines Treffers fehlt
    false_rim_make_prob: float = 0.0  # Fehlwurf bekommt ein falsches RimMake
    three_prob_noise: float = 0.0  # sigma (m) Positionsfehler der ShotRelease-Position
    possession_jitter: float = 0.0  # sigma (s) Zeitjitter der Possession-Signale
    drop_shot_prob: float = 0.0  # ShotRelease fehlt komplett

    def is_clean(self) -> bool:
        return not any((self.drop_player_id_prob, self.miss_rim_make_prob, self.false_rim_make_prob,
                        self.three_prob_noise, self.possession_jitter, self.drop_shot_prob))


@dataclass
class SimResult:
    truth: list[Event]
    observations: list[Observation]
    scoreboard: list[tuple[float, dict]]
    final_score: dict[str, int]
    track_of: dict[str, int]  # player_id -> erste Track-ID
    track_stints: dict[str, list[int]]  # player_id -> Track-IDs der Feld-Einsaetze
    points: dict[str, int] = field(default_factory=dict)
    zones: dict[str, list[tuple[float, float]]] = field(default_factory=default_zones)
    duration: float = 0.0  # video_ts am Ende
    seed: int = 0
    periods_played: int = 0


class _PeriodEnd(Exception):
    pass


# Angriffsformation: (Abstand zur Grundlinie, y)
OFF_SLOTS = np.array([(8.2, 0.0), (6.0, -5.4), (6.0, 5.4), (1.8, -6.2), (3.2, 3.0)])
BENCH_Y = -9.2
SUB_ENTRY = (0.0, -7.9)
CLIP_X, CLIP_Y = ct.HALF_L - 0.3, ct.HALF_W - 0.3


class _Sim:
    def __init__(self, seed, game, rosters, periods, period_length_s, noise, frame_hz, include_frames,
                 referees, break_s, halftime_s, tech_foul_prob):
        self.seed = seed
        self.game = game
        self.rng = random.Random(seed)
        self.nrng = random.Random(seed * 7919 + 13)
        self.frng = np.random.default_rng(seed + 1_000_003)
        self.noise = noise or SimNoise()
        self.hz = float(frame_hz)
        self.frames = bool(include_frames)
        self.periods = periods
        self.plen = float(period_length_s)
        self.otlen = float(game.overtime_length_s)
        self.break_s = break_s
        self.halftime_s = halftime_s
        self.tech_p = tech_foul_prob
        self.nref = referees

        self.players: dict[str, Player] = {}
        self.lineup: dict[str, list[str]] = {}
        self.bench: dict[str, list[str]] = {}
        self.skill: dict[str, dict[str, float]] = {}
        for side in SIDES:
            roster = rosters.get(side)
            if roster is None:
                roster = rosters.get(game.team_id(side))
            if roster is None or len(roster) < 5:
                raise ValueError(f"roster for {side} needs at least 5 players")
            self.lineup[side] = [p.id for p in roster[:5]]
            self.bench[side] = [p.id for p in roster[5:]]
            for k, p in enumerate(roster):
                self.players[p.id] = p
                self.skill[p.id] = dict(
                    use=(0.7 + 0.6 * self.rng.random()) * (1.0 if k < 5 else 0.55),
                    p2=self.rng.gauss(0, 0.04), p3=self.rng.gauss(0, 0.05),
                    ft=self.rng.gauss(0, 0.07), foul=0.5 + self.rng.random(),
                )
        self.pf = {pid: 0 for pid in self.players}  # persoenliche Fouls
        self.out = {pid: False for pid in self.players}  # ausgeschieden (5 Fouls)
        self.points = {pid: 0 for pid in self.players}

        # tracks
        self.next_track = 1
        self.track: dict[str, int] = {}
        self.track_of: dict[str, int] = {}
        self.track_stints: dict[str, list[int]] = {pid: [] for pid in self.players}
        for side in SIDES:
            for pid in self.lineup[side]:
                self.track[pid] = self._new_track()
                self.track_stints[pid].append(self.track[pid])
        for side in SIDES:
            for pid in self.bench[side]:
                self.track[pid] = self._new_track()
        self.track_of = dict(self.track)

        # time / clock
        self.t = 0.0
        self.period = 1
        self.clock = self.plen
        self.running = False
        self.expired = False
        self.expiry_t = 0.0
        self._run_start = 0.0
        self._run_clock0 = 0.0
        self._k = 1
        self.run_acc = 0.0
        self.entered = {pid: 0.0 for s in SIDES for pid in self.lineup[s]}
        self.left = {pid: 0.0 for s in SIDES for pid in self.bench[s]}
        self._fi = 0
        self._sb = 0

        # game state
        self.score = {"home": 0, "away": 0}
        self.disp = {"home": 0, "away": 0}
        self.pending: list[tuple[float, str, int]] = []
        self.tf = {"home": 0, "away": 0}
        self.timeouts = {"home": 2, "away": 2}
        self.off: Side = "home"
        self.mode = "dead"
        self.start_idx: Optional[int] = None
        self.fast = False
        self.arrow: Side = "away"
        self.holder: Optional[tuple[str, int]] = None
        self.last_passer: Optional[int] = None
        self.sc_t0 = 0.0
        self.n_ev = 0
        self.truth: list[Event] = []
        self.obs: list[Observation] = []
        self.scoreboard: list[tuple[float, dict]] = []

        # geometry / motion
        self.cpos = np.zeros((2, 5, 2))
        self.ctgt = np.zeros((2, 5, 2))
        self.jit = np.zeros((2, 5, 2))
        self.bpos = {s: np.zeros((len(self.bench[s]), 2)) for s in SIDES}
        self.btgt = {s: np.zeros((len(self.bench[s]), 2)) for s in SIDES}
        for s in SIDES:
            nb = len(self.bench[s])
            for j in range(nb):
                x = (-9.2 + 6.6 * j / max(nb - 1, 1)) if s == "home" else (2.6 + 6.6 * j / max(nb - 1, 1))
                self.btgt[s][j] = (x, BENCH_Y)
            self.bpos[s][:] = self.btgt[s]
        self.ball: tuple = ("free", (0.0, 0.0))
        self.refpos = np.zeros((referees, 2))
        for i in range(referees):
            self.refpos[i] = (-3.0 + 3.0 * i, 8.2 if i % 2 else -7.8)

    # ------------------------------------------------------------------ utils
    def _new_track(self) -> int:
        self.next_track += 1
        return self.next_track - 1

    def rnd(self, a=None, b=None):
        return self.rng.random() if a is None else self.rng.uniform(a, b)

    def sx(self, side: Side) -> int:
        """+1: Seite greift den rechten Korb an."""
        eff = min(self.period, self.periods)
        home_right = eff <= self.periods / 2
        return 1 if (side == "home") == home_right else -1

    @staticmethod
    def basket_of(sx: int) -> str:
        return "right" if sx > 0 else "left"

    def pid(self, side, idx) -> str:
        return self.lineup[side][idx]

    def _pick(self, side, exclude=(), key="use") -> int:
        idxs = [i for i in range(5) if i not in exclude]
        w = [self.skill[self.lineup[side][i]][key] for i in idxs]
        return self.rng.choices(idxs, w)[0]

    # ------------------------------------------------------------------ emit
    def _ev(self, type_, team=None, player=None, payload=None, ts=None, clock=None) -> Event:
        self.n_ev += 1
        e = Event(
            id=f"{self.game.id}-s{self.seed}-{self.n_ev:05d}", game_id=self.game.id, type=type_,
            video_ts=round(self.t if ts is None else ts, 3), period=self.period,
            clock=round(self.clock if clock is None else clock, 1), team=team, player_id=player,
            payload=payload or {}, confidence=1.0, source=EventSource.CONSOLE,
            status=EventStatus.CONFIRMED,
        )
        self.truth.append(e)
        return e

    def _pobs(self, pid):
        p = self.noise.drop_player_id_prob
        if p > 0 and self.nrng.random() < p:
            return None
        return pid

    def _emit_clock(self, clock, running):
        self.obs.append(ClockState(self.t, self.period, clock, running))

    def _set_holder(self, side, idx, passer=None):
        self.holder = (side, idx)
        self.last_passer = passer
        self.ball = ("held", side, idx)
        pid = self.pid(side, idx)
        ts = self.t
        j = self.noise.possession_jitter
        if j > 0:
            ts = max(0.0, ts + self.nrng.gauss(0, j))
        self.obs.append(Possession(ts, self.track[pid], side, self._pobs(pid)))

    def _free_ball(self):
        x, y, _ = self._ball_xyz(self.t)
        self.ball = ("free", (x, y))
        self.holder = None
        j = self.noise.possession_jitter
        ts = max(0.0, self.t + self.nrng.gauss(0, j)) if j > 0 else self.t
        self.obs.append(Possession(ts, None, None, None))

    # ------------------------------------------------------------------ clock
    def _stop_clock(self):
        if self.running:
            self.running = False
            self.clock = max(0.1, round(self.clock, 1)) if self.clock > 0 else 0.0
            self._emit_clock(self.clock, False)

    def _start_clock(self):
        if not self.running and self.clock > 0:
            self.running = True
            self._run_start = self.t
            self._run_clock0 = self.clock
            self._k = 1
            self._emit_clock(self.clock, True)

    def advance(self, dt: float, soft: bool = False):
        T = self.t + dt
        if self.running:
            tz = self._run_start + self._run_clock0
            if T >= tz - 1e-9:
                self._run_to(max(tz, self.t))
                self.clock = 0.0
                self.running = False
                self.expired = True
                self.expiry_t = self.t
                self._emit_clock(0.0, False)
                if not soft:
                    raise _PeriodEnd()
        self._run_to(T)

    def _check_expired(self):
        if self.expired:
            raise _PeriodEnd()

    def _run_to(self, T: float):
        eps = 1e-9
        while True:
            nf = self._fi / self.hz if self.frames else INF
            ns = float(self._sb)
            nk = (self._run_start + self._k) if self.running else INF
            nxt = min(T, nf, ns, nk)
            if nxt > self.t:
                if self.running:
                    self.run_acc += nxt - self.t
                self.t = nxt
                if self.running:
                    self.clock = self._run_clock0 - (self.t - self._run_start)
            if nf <= nxt + eps:
                self._frame(self.t)
                self._fi += 1
            if ns <= nxt + eps:
                self._sb_tick()
                self._sb += 1
            if nk <= nxt + eps and self.running:
                val = self._run_clock0 - self._k
                if val > eps:
                    self._emit_clock(round(val, 1), True)
                self._k += 1
            if nxt >= T - eps:
                break

    def _sb_tick(self):
        if self.pending:
            keep = []
            for it in self.pending:
                if it[0] <= self.t + 1e-9:
                    self.disp[it[1]] += it[2]
                else:
                    keep.append(it)
            self.pending = keep
        self.scoreboard.append((self.t, {
            "home": self.disp["home"], "away": self.disp["away"], "clock": round(self.clock, 1),
            "period": self.period, "running": self.running,
            "fouls_home": self.tf["home"], "fouls_away": self.tf["away"],
        }))

    # ------------------------------------------------------------------ frames
    def _ball_xyz(self, ts):
        b = self.ball
        k = b[0]
        if k == "held":
            p = self.cpos[SIDE_IDX[b[1]], b[2]]
            return float(p[0]), float(p[1]), 1.0
        if k == "pass":
            _, side, i0, i1, t0, t1 = b
            u = min(1.0, max(0.0, (ts - t0) / (t1 - t0)))
            a = self.cpos[SIDE_IDX[side], i0]
            c = self.cpos[SIDE_IDX[side], i1]
            p = a + (c - a) * u
            return float(p[0]), float(p[1]), 1.2 + 0.6 * math.sin(math.pi * u)
        if k == "shot":
            _, x0, y0, bx, by, t0, t1 = b
            u = min(1.0, max(0.0, (ts - t0) / (t1 - t0)))
            h = 2.2 * (1 - u) + 3.05 * u + 4 * 1.3 * u * (1 - u)
            return x0 + (bx - x0) * u, y0 + (by - y0) * u, h
        x, y = b[1]
        return x, y, 0.15

    def _frame(self, ts):
        dt = 1.0 / self.hz
        fr = self.frng
        if self._fi % 15 == 0:
            self.jit = fr.normal(0.0, 0.45, (2, 5, 2))
        tgt = self.ctgt + self.jit
        tgt[..., 0] = np.clip(tgt[..., 0], -CLIP_X, CLIP_X)
        tgt[..., 1] = np.clip(tgt[..., 1], -CLIP_Y, CLIP_Y)
        d = tgt - self.cpos
        dist = np.hypot(d[..., 0], d[..., 1])
        spd = 5.0 * dt * (0.8 + 0.2 * fr.random((2, 5)))
        scale = np.minimum(1.0, spd / np.maximum(dist, 1e-9))
        self.cpos += d * scale[..., None]
        for s in SIDES:
            if len(self.bench[s]):
                db = self.btgt[s] - self.bpos[s]
                dd = np.maximum(np.hypot(db[:, 0], db[:, 1]), 1e-9)
                self.bpos[s] += db * np.minimum(1.0, 3.0 * dt / dd)[:, None]
        bx, by, bh = self._ball_xyz(ts)
        meas = (self.cpos + fr.normal(0.0, 0.03, (2, 5, 2))).tolist()
        dropp = self.noise.drop_player_id_prob
        players = []
        for s in SIDES:
            si = SIDE_IDX[s]
            lu = self.lineup[s]
            for i in range(5):
                pid = lu[i]
                conf = 0.95
                if dropp > 0 and self.nrng.random() < dropp:
                    pid, conf = None, 0.0
                x, y = meas[si][i]
                players.append(PlayerState(self.track[lu[i]], s, x, y, pid, conf))
            bp = self.bpos[s].tolist()
            for j, bid in enumerate(self.bench[s]):
                pid = bid
                conf = 0.9
                if dropp > 0 and self.nrng.random() < dropp:
                    pid, conf = None, 0.0
                players.append(PlayerState(self.track[bid], s, bp[j][0], bp[j][1], pid, conf))
        if self.nref:
            tx, ty = bx, by
            for i in range(self.nref):
                off = (-4.0 if i % 2 else 4.0, 2.5 if i % 2 else -2.5)
                tgt_r = np.array([tx + off[0] * 0.5, ty + off[1]])
                tgt_r[0] = np.clip(tgt_r[0], -CLIP_X, CLIP_X)
                tgt_r[1] = np.clip(tgt_r[1], -CLIP_Y, CLIP_Y)
                dv = tgt_r - self.refpos[i]
                n = max(float(np.hypot(*dv)), 1e-9)
                self.refpos[i] += dv * min(1.0, 3.5 * dt / n)
                players.append(PlayerState(9000 + i, None, float(self.refpos[i][0]), float(self.refpos[i][1]), None, 0.0))
        self.obs.append(FrameState(ts, tuple(players), bx, by, bh))

    # ------------------------------------------------------------------ layouts
    def _layout_half(self, off: Side):
        sx = self.sx(off)
        oi = SIDE_IDX[off]
        xy = np.stack([sx * (ct.HALF_L - OFF_SLOTS[:, 0]), OFF_SLOTS[:, 1]], axis=1)
        self.ctgt[oi] = xy
        basket = np.array([sx * ct.BASKET_X, 0.0])
        vec = basket - xy
        n = np.maximum(np.hypot(vec[:, 0], vec[:, 1]), 1e-9)[:, None]
        self.ctgt[1 - oi] = xy + 1.3 * vec / n

    def _layout_reb(self, off: Side):
        sx = self.sx(off)
        oi = SIDE_IDX[off]
        bx = sx * ct.BASKET_X
        for row, r, angs in ((oi, 3.3, (-80, -40, 0, 40, 80)), (1 - oi, 1.9, (-60, -30, 0, 30, 60))):
            for i, a in enumerate(angs):
                a = math.radians(a)
                self.ctgt[row, i] = (bx - sx * r * math.cos(a), r * math.sin(a))

    def _layout_ft(self, side: Side, shooter: int):
        sx = self.sx(side)
        si = SIDE_IDX[side]
        fx = sx * (ct.HALF_L - ct.FT_LINE_FROM_BASELINE)
        mine = [(2.8, -2.9), (2.8, 2.9), (4.6, -2.9), (4.6, 2.9)]
        theirs = [(1.9, -2.0), (1.9, 2.0), (3.7, -2.0), (3.7, 2.0), (8.0, 0.0)]
        k = 0
        for i in range(5):
            if i == shooter:
                self.ctgt[si, i] = (fx, 0.0)
            else:
                d, y = mine[k]
                k += 1
                self.ctgt[si, i] = (sx * (ct.HALF_L - d), y)
        for i, (d, y) in enumerate(theirs):
            self.ctgt[1 - si, i] = (sx * (ct.HALF_L - d), y)

    def _layout_huddle(self):
        for s, cx in (("home", -4.0), ("away", 4.0)):
            for i in range(5):
                a = 2 * math.pi * i / 5
                self.ctgt[SIDE_IDX[s], i] = (cx + 1.0 * math.cos(a), -6.0 + 0.8 * math.sin(a))

    def _layout_jump(self):
        for s, sg in (("home", -1), ("away", 1)):
            si = SIDE_IDX[s]
            self.ctgt[si, 0] = (sg * 0.6, 0.0)
            for i in range(1, 5):
                a = math.radians(-60 + 40 * (i - 1))
                self.ctgt[si, i] = (sg * 3.2 * math.cos(a) * 0.9, 3.2 * math.sin(a))

    def _snap(self):
        self.cpos[:] = self.ctgt

    # ------------------------------------------------------------------ shots
    def _tp(self, x, y, basket):
        return ct.three_probability(x, y, basket)

    def _shot_loc(self, value: int, sx: int, rim_only: bool) -> tuple[float, float]:
        rng = self.rng
        bx = sx * ct.BASKET_X
        basket = self.basket_of(sx)
        for _ in range(300):
            if value == 3:
                if rng.random() < 0.25:
                    x = sx * rng.uniform(11.5, 13.5)
                    y = rng.choice((-1, 1)) * rng.uniform(7.0, 7.35)
                else:
                    r = ct.THREE_RADIUS + rng.uniform(0.45, 1.6)
                    a = math.radians(rng.uniform(-80, 80))
                    x, y = bx - sx * r * math.cos(a), r * math.sin(a)
                ok = ct.in_court(x, y, -0.1) and self._tp(x, y, basket) > 0.95
            else:
                if rim_only or rng.random() < 0.5:
                    r = rng.uniform(0.4, 1.5)
                    a = math.radians(rng.uniform(-100, 100))
                else:
                    r = rng.uniform(2.0, 6.0)
                    a = math.radians(rng.uniform(-85, 85))
                x, y = bx - sx * r * math.cos(a), r * math.sin(a)
                ok = ct.in_court(x, y, -0.1) and self._tp(x, y, basket) < 0.05
            if ok:
                return x, y
        return (bx - sx * (8.0 if value == 3 else 2.0), 0.0)

    def _emit_release(self, side, idx, x, y, basket, tp_override=None):
        n = self.noise
        if n.drop_shot_prob > 0 and self.nrng.random() < n.drop_shot_prob:
            return
        xo, yo = x, y
        if n.three_prob_noise > 0:
            xo += self.nrng.gauss(0, n.three_prob_noise)
            yo += self.nrng.gauss(0, n.three_prob_noise)
        tp = self._tp(xo, yo, basket)
        pid = self.pid(side, idx)
        self.obs.append(ShotRelease(self.t, self.track[pid], side, xo, yo, basket, self._pobs(pid), tp, 1.0))

    def _emit_rim(self, ts, basket, made, flight):
        n = self.noise
        if made:
            if n.miss_rim_make_prob > 0 and self.nrng.random() < n.miss_rim_make_prob:
                return
            self.obs.append(RimMake(ts, basket, 1.0))
        elif n.false_rim_make_prob > 0 and self.nrng.random() < n.false_rim_make_prob:
            self.obs.append(RimMake(ts, basket, 0.6))

    def _score(self, side, pid, pts):
        self.score[side] += pts
        self.points[pid] += pts
        self.pending.append((self.t + self.rnd(1.0, 3.0), side, pts))

    # ------------------------------------------------------------------ dead balls
    def _foul(self, side, idx, kind: FoulKind, fouled_pid, count_team=True):
        pid = self.pid(side, idx)
        self._ev(EventType.FOUL, side, pid, {"kind": kind.value, "fouled_id": fouled_pid})
        if kind != FoulKind.TECHNICAL:
            self.pf[pid] += 1
            if self.pf[pid] >= 5:
                self.out[pid] = True
            self.tf[side] += 1

    def _dead_ball(self, protect=frozenset()):
        self._stop_clock()
        self.ball = ("free", self._ball_xyz(self.t)[:2])
        self.holder = None
        saved = (self.off, self.mode, self.start_idx, self.fast)
        # technical foul (rare)
        if self.rng.random() < self.tech_p:
            side = self.rng.choice(SIDES)
            idx = self._pick(side, key="foul")
            fp = self.pid(side, idx)
            self.advance(self.rnd(0.5, 1.0))
            self._ev(EventType.FOUL, side, fp, {"kind": FoulKind.TECHNICAL.value, "fouled_id": None})
            other = other_side(side)
            self.advance(self.rnd(0.5, 1.0))
            self._free_throws(other, self._pick(other), 1, resume=saved[0])
        self.off, self.mode, self.start_idx, self.fast = saved
        boost = 1.0
        cand = [s for s in SIDES if self.timeouts[s] > 0]
        if cand and self.rng.random() < 0.06:
            side = self.rng.choice(cand)
            self.timeouts[side] -= 1
            self.advance(self.rnd(0.5, 1.0))
            self._ev(EventType.TIMEOUT, side)
            self._layout_huddle()
            self.advance(self.rnd(1.5, 2.5))
            boost = 4.0
            self._subs(protect, boost)
            self.advance(58.0)
            return
        self._subs(protect, boost)

    def _subs(self, protect, boost=1.0):
        for side in SIDES:
            lu = self.lineup[side]
            be = self.bench[side]
            avail = [j for j, p in enumerate(be) if not self.out[p]]
            if not avail:
                continue
            outs = [i for i, p in enumerate(lu) if self.out[p]]
            stints = [self.run_acc - self.entered[p] for p in lu]
            mx = max(stints)
            p = (0.06 + (0.15 if mx > 300 else 0.0) + (0.35 if mx > 420 else 0.0)) * boost
            if self.rng.random() < min(0.95, p):
                k = self.rng.choice((1, 1, 1, 2, 2, 3))
                cands = [i for i in range(5) if i not in outs and lu[i] not in protect]
                w = [math.exp(stints[i] / 200.0) for i in cands]
                while k > 0 and cands:
                    i = self.rng.choices(cands, w)[0]
                    w.pop(cands.index(i))
                    cands.remove(i)
                    outs.append(i)
                    k -= 1
            outs = outs[: len(avail)]
            if not outs:
                continue
            self.advance(self.rnd(0.8, 1.5))
            for i in outs:
                avail = [j for j, p in enumerate(be) if not self.out[p]]
                if not avail:
                    break
                w = [self.skill[be[j]]["use"] * math.exp((self.run_acc - self.left[be[j]]) / 400.0) for j in avail]
                j = self.rng.choices(avail, w)[0]
                self._swap(side, i, j)
                self.advance(self.rnd(0.3, 0.8))

    def _swap(self, side, i, j):
        si = SIDE_IDX[side]
        lu, be = self.lineup[side], self.bench[side]
        pout, pin = lu[i], be[j]
        self._ev(EventType.SUBSTITUTION, side, pin, {"player_in": pin, "player_out": pout})
        out_xy = self.cpos[si, i].copy()
        self.bpos[side][j] = out_xy
        lu[i], be[j] = pin, pout
        self.entered[pin] = self.run_acc
        self.left[pout] = self.run_acc
        self.track[pin] = self._new_track()
        self.track_stints[pin].append(self.track[pin])
        self.track[pout] = self._new_track()
        self.cpos[si, i] = SUB_ENTRY
        if self.holder == (side, i):
            self.holder = None

    # ------------------------------------------------------------------ free throws
    def _free_throws(self, side: Side, idx: int, n: int, resume: Optional[Side] = None):
        """n Freiwuerfe (Uhr steht). resume: danach Einwurf fuer diese Seite, ohne Rebound."""
        sx = self.sx(side)
        basket = self.basket_of(sx)
        pid = self.pid(side, idx)
        self._layout_ft(side, idx)
        self.advance(self.rnd(1.5, 3.0))
        pt = self.track[pid]
        self.obs.append(Formation(self.t, "free_throw", pt, side, self._pobs(pid), basket))
        fx = sx * (ct.HALF_L - ct.FT_LINE_FROM_BASELINE)
        made = False
        p = min(0.95, max(0.4, 0.70 + self.skill[pid]["ft"]))
        for k in range(1, n + 1):
            self.advance(self.rnd(2.0, 3.0))
            self.cpos[SIDE_IDX[side], idx] = (fx, 0.0)
            made = self.rng.random() < p
            flight = self.rnd(0.6, 1.0)
            self._ev(EventType.FREE_THROW, side, pid, {"made": made, "attempt": k, "of": n})
            self._set_holder(side, idx)
            self._emit_release(side, idx, fx, 0.0, basket)
            bx = sx * ct.BASKET_X
            self.ball = ("shot", fx, 0.0, bx, 0.0, self.t, self.t + flight)
            self.advance(flight)
            if made:
                self._score(side, pid, 1)
            self._emit_rim(self.t, basket, made, flight)
            if k < n:
                self.ball = ("free", (bx, 0.0))
        if resume is not None:
            self.off, self.mode, self.start_idx, self.fast = resume, "dead", None, False
            self.ball = ("free", (0.0, -7.5))
            self.holder = None
        elif made:
            self.off, self.mode, self.start_idx, self.fast = other_side(side), "dead", None, False
            self.ball = ("free", (sx * (ct.HALF_L + 0.6), 0.0))
            self.holder = None
        else:
            self._miss_rebound(side, idx, blocked=False, oob_p=0.02, ft=True)

    # ------------------------------------------------------------------ rebounds
    def _miss_rebound(self, off: Side, shooter: int, blocked: bool, oob_p: float, ft: bool = False):
        de = other_side(off)
        sx = self.sx(off)
        basket = self.basket_of(sx)
        bx = sx * ct.BASKET_X
        self._layout_reb(off)
        self.ball = ("free", (bx, 0.0))
        if self.holder is not None:
            self._free_ball()
        if self.rng.random() < oob_p:
            self.advance(self.rnd(0.6, 1.2))
            off_touch = self.rng.random() < 0.5
            if off_touch:
                lt_side, lt_idx = off, shooter
            else:
                lt_side, lt_idx = de, self._pick(de)
            pid = self.pid(lt_side, lt_idx)
            self.obs.append(OutOfBounds(self.t, self.track[pid], lt_side, self._pobs(pid)))
            self._stop_clock()
            recv = other_side(lt_side)
            self._ev(EventType.REBOUND, recv, None, {"offensive": recv == off})
            self.off, self.mode, self.start_idx, self.fast = recv, "dead", None, False
            self._dead_ball()
            return
        self.advance(self.rnd(0.5, 1.0))
        orb = self.rng.random() < (0.30 if blocked else (0.18 if ft else 0.25))
        side = off if orb else de
        idx = self._pick(side)
        pid = self.pid(side, idx)
        a = self.rnd(0, 2 * math.pi)
        r = self.rnd(1.2, 3.0)
        self.cpos[SIDE_IDX[side], idx] = (bx - sx * r * abs(math.cos(a)), r * math.sin(a))
        self._ev(EventType.REBOUND, side, pid, {"offensive": orb})
        self._set_holder(side, idx)
        self._start_clock()
        self.off, self.start_idx, self.fast = side, idx, False
        self.mode = ("putback" if self.rng.random() < 0.6 else "orb") if orb else "carry"

    # ------------------------------------------------------------------ passing
    def _pass(self, side, hold):
        i0 = self.holder[1]
        i1 = self._pick(side, exclude=(i0,))
        self.advance(hold)
        self.ball = ("pass", side, i0, i1, self.t, self.t + 0.5)
        self._free_ball_keep()
        self.advance(0.5)
        self._set_holder(side, i1, passer=i0)

    def _free_ball_keep(self):
        """Ball ist im Flug: Possession(None), Ballmodus bleibt 'pass'."""
        self.holder = None
        j = self.noise.possession_jitter
        ts = max(0.0, self.t + self.nrng.gauss(0, j)) if j > 0 else self.t
        self.obs.append(Possession(ts, None, None, None))

    def _passes(self, side, n):
        for _ in range(n):
            self._pass(side, self.rnd(1.6, 2.8))

    # ------------------------------------------------------------------ possession
    def _possession(self):
        self._check_expired()
        rng = self.rng
        off = self.off
        de = other_side(off)
        oi = SIDE_IDX[off]
        sx = self.sx(off)
        basket = self.basket_of(sx)
        mode = self.mode
        idx = self.start_idx
        fast = self.fast
        self.fast = False
        if mode in ("orb", "putback"):
            pass
        else:
            self._layout_half(off)
        if mode == "dead":
            if idx is None:
                idx = self._pick(off)
            self.advance(self.rnd(1.0, 2.0))
            # Einwurf an der Seitenlinie
            self.cpos[oi, idx] = (self.rnd(-8, 8), rng.choice((-1, 1)) * (ct.HALF_W + 0.5))
            self._set_holder(off, idx)
            self._start_clock()
            self.sc_t0 = self.t
            self.advance(self.rnd(1.5, 3.0))
        elif mode == "inb_live":
            idx = self._pick(off)
            self.advance(self.rnd(0.8, 1.5))
            self.cpos[oi, idx] = (-sx * (ct.HALF_L + 0.6), 0.0)
            self._set_holder(off, idx)
            self.sc_t0 = self.t
            self.advance(self.rnd(4.5, 7.0))
        elif mode == "carry":
            self.sc_t0 = self.t
            self.advance(self.rnd(2.0, 3.5) if fast else self.rnd(4.5, 7.0))
        elif mode == "orb":
            self.sc_t0 = self.t
            self._layout_half(off)
            self.advance(self.rnd(0.3, 0.8))
        else:  # putback
            self.sc_t0 = self.t

        # outcome
        r = rng.random()
        if mode == "putback":
            outcome = "shootfoul" if r < 0.10 else "shot"
        else:
            outcome = "shot"
            acc = 0.0
            for name, p in (("steal", 0.07), ("oob", 0.035), ("travel", 0.02),
                            ("shotclock", 0.012 if mode in ("dead", "inb_live", "carry") else 0.0),
                            ("offfoul", 0.02), ("nsfoul", 0.075), ("shootfoul", 0.075)):
                acc += p
                if r < acc:
                    outcome = name
                    break
        if mode == "putback":
            npass = 0
        elif mode == "orb":
            npass = rng.choice((0, 1, 1, 2))
        elif fast:
            npass = rng.choice((0, 0, 1, 2))
        else:
            npass = rng.choices((1, 2, 3, 4), (0.2, 0.35, 0.3, 0.15))[0]
        if outcome in ("steal", "oob", "travel", "offfoul", "nsfoul"):
            npass = rng.randint(0, min(npass, 3))

        if outcome == "shotclock":
            while (self.t - self.sc_t0) + 3.0 < 24.0:
                self._pass(off, self.rnd(1.6, 2.8))
            self.advance(max(0.0, self.sc_t0 + 24.0 - self.t))
            self._stop_clock()
            self._ev(EventType.TURNOVER, off, None, {"kind": TurnoverKind.SHOT_CLOCK.value})
            self.off, self.mode, self.start_idx = de, "dead", None
            self._dead_ball()
            return

        self._passes(off, npass)
        cur = self.holder[1]
        cpid = self.pid(off, cur)

        if outcome == "steal":
            thief = self._pick(de)
            tp = self.pid(de, thief)
            self.advance(self.rnd(1.0, 2.0))
            self.ball = ("pass", off, cur, self._pick(off, exclude=(cur,)), self.t, self.t + 0.5)
            self._free_ball_keep()
            self.advance(0.25)
            self.obs.append(BallTouch(self.t, self.track[tp], de, self._pobs(tp), False))
            self._ev(EventType.STEAL, de, tp, {"victim_id": cpid})
            self._ev(EventType.TURNOVER, off, cpid, {"kind": TurnoverKind.STEAL.value})
            hx, hy, _ = self._ball_xyz(self.t)
            self.cpos[SIDE_IDX[de], thief] = (hx, hy)
            self.advance(0.3)
            self._set_holder(de, thief)
            self.off, self.mode, self.start_idx, self.fast = de, "carry", thief, True
            self._check_expired()
            return
        if outcome == "oob":
            self.advance(self.rnd(1.0, 2.0))
            self.obs.append(OutOfBounds(self.t, self.track[cpid], off, self._pobs(cpid)))
            self._stop_clock()
            self._ev(EventType.TURNOVER, off, cpid, {"kind": TurnoverKind.OUT_OF_BOUNDS.value})
            self.off, self.mode, self.start_idx = de, "dead", None
            self._dead_ball()
            return
        if outcome == "travel":
            self.advance(self.rnd(1.0, 2.0))
            self._stop_clock()
            self._ev(EventType.TURNOVER, off, cpid, {"kind": TurnoverKind.TRAVEL.value})
            self.off, self.mode, self.start_idx = de, "dead", None
            self._dead_ball()
            return
        if outcome == "offfoul":
            self.advance(self.rnd(1.0, 2.0))
            self._stop_clock()
            dpid = self.pid(de, self._pick(de, key="foul"))
            self._foul(off, cur, FoulKind.OFFENSIVE, dpid)
            self._ev(EventType.TURNOVER, off, cpid, {"kind": TurnoverKind.OFFENSIVE_FOUL.value})
            self.off, self.mode, self.start_idx = de, "dead", None
            self._dead_ball()
            return
        if outcome == "nsfoul":
            self.advance(self.rnd(1.0, 2.0))
            self._stop_clock()
            fouler = self._pick(de, key="foul")
            fouled = cur if rng.random() < 0.5 else self._pick(off)
            fpid = self.pid(off, fouled)
            self._foul(de, fouler, FoulKind.PERSONAL, fpid)
            bonus = self.tf[de] >= 5
            self.off, self.mode, self.start_idx = off, "dead", None
            self._dead_ball(protect={fpid} if bonus else frozenset())
            if bonus:
                self._free_throws(off, fouled, 2)
            return

        # shot or shooting foul
        p3 = 0.0 if mode == "putback" else 0.36
        value = 3 if rng.random() < p3 else 2
        rim_only = mode == "putback"
        sx_, sy_ = self._shot_loc(value, sx, rim_only)
        shooter = cur
        spid = self.pid(off, shooter)
        self.ctgt[oi, shooter] = (sx_, sy_)
        hold = self.rnd(1.6, 2.8) if mode != "putback" else self.rnd(0.6, 1.4)
        if outcome == "shootfoul":
            self.advance(hold)
            self._stop_clock()
            fouler = self._pick(de, key="foul")
            self._foul(de, fouler, FoulKind.PERSONAL, spid)
            self.off, self.mode, self.start_idx = off, "dead", None
            self._dead_ball(protect={spid})
            self._free_throws(off, shooter, value)
            return
        self._take_shot(off, shooter, value, sx_, sy_, hold, mode)

    def _take_shot(self, off, shooter, value, x, y, hold, mode):
        rng = self.rng
        de = other_side(off)
        oi = SIDE_IDX[off]
        sx = self.sx(off)
        basket = self.basket_of(sx)
        bx, by = sx * ct.BASKET_X, 0.0
        spid = self.pid(off, shooter)
        sk = self.skill[spid]
        rim = ct.distance_to_basket(x, y, basket) < 2.0
        assister = self.last_passer
        p_blk = 0.02 if value == 3 else (0.08 if rim else 0.05)
        blocked = rng.random() < p_blk
        if value == 3:
            pm = 0.33 + sk["p3"]
        else:
            pm = (0.58 if rim else 0.40) + sk["p2"]
        made = (not blocked) and rng.random() < pm
        and1 = made and rng.random() < 0.045
        # Assist-Wahrheit wie beim Statistiker: Pass und Abschluss ohne lange
        # Eigenaktion (Ballhalten <= ASSIST_MAX_HOLD_S). Beobachtbar über die
        # Ballbesitzkette, deshalb nicht zufällig.
        assisted = made and assister is not None and hold <= ASSIST_MAX_HOLD_S
        flight = rng.uniform(0.6, 1.0)
        blocker = self._pick(de) if blocked else None
        fouler = self._pick(de, key="foul") if and1 else None
        pid_assist = self.pid(off, assister) if assisted else None

        self.advance(hold)
        self.cpos[oi, shooter] = (x, y)
        shot_ev = self._ev(EventType.SHOT, off, spid, {"value": value, "made": made, "x": round(x, 3), "y": round(y, 3),
                                                       "three_prob": round(self._tp(x, y, basket), 4)})
        t_rel = self.t
        self._emit_release(off, shooter, x, y, basket)
        self.ball = ("shot", x, y, bx, by, t_rel, t_rel + flight)
        self._free_ball_keep()
        self.holder = None
        self._layout_reb(off)
        if blocked:
            self.advance(0.25, soft=True)
            bpid = self.pid(de, blocker)
            self.obs.append(BallTouch(self.t, self.track[bpid], de, self._pobs(bpid), True))
            self._ev(EventType.BLOCK, de, bpid, {"shot_id": shot_ev.id, "shooter_id": spid})
            self._emit_rim(t_rel + flight, basket, False, flight)
            self.ball = ("free", self._ball_xyz(self.t)[:2])
            self._check_expired()
            self._miss_rebound(off, shooter, blocked=True, oob_p=0.18)
            return
        self.advance(flight, soft=True)
        self._emit_rim(t_rel + flight, basket, made, flight)
        if made:
            self._score(off, spid, value)
            if assisted:
                self._ev(EventType.ASSIST, off, pid_assist, {"shot_id": shot_ev.id})
            self.ball = ("free", (bx, by))
            if and1:
                self.advance(0.4, soft=True)
                self._stop_clock()
                self._foul(de, fouler, FoulKind.PERSONAL, spid)
                self._check_expired()
                self.off, self.mode, self.start_idx = off, "dead", None
                self._dead_ball(protect={spid})
                self._free_throws(off, shooter, 1)
                return
            self._check_expired()
            if self.period >= self.periods and self.clock <= 120.0:
                self._stop_clock()
                self.off, self.mode, self.start_idx, self.fast = de, "dead", None, False
                self._dead_ball()
            else:
                self.off, self.mode, self.start_idx, self.fast = de, "inb_live", None, False
            return
        self._check_expired()
        self._miss_rebound(off, shooter, blocked=False, oob_p=0.06)

    # ------------------------------------------------------------------ game flow
    def _play_period(self, p: int, length: float):
        self.period = p
        self.clock = length
        self.expired = False
        self.running = False
        self.tf = {"home": 0, "away": 0}
        if p <= self.periods:
            if p == self.periods // 2 + 1:
                self.timeouts = {"home": 3, "away": 3}
        else:
            self.timeouts = {"home": 1, "away": 1}
        self._emit_clock(self.clock, False)
        self._ev(EventType.PERIOD_START)
        if p == 1:
            self._jump_ball()
        else:
            self._layout_half(self.arrow)
            self.advance(4.0)
            self.off, self.mode, self.start_idx, self.fast = self.arrow, "dead", None, False
            self.arrow = other_side(self.arrow)
            self._subs(frozenset(), 2.5)
        try:
            while True:
                self._possession()
        except _PeriodEnd:
            pass
        self._stop_clock()
        self._ev(EventType.PERIOD_END, ts=self.expiry_t, clock=0.0)
        self.ball = ("free", (0.0, 0.0))
        self.holder = None

    def _jump_ball(self):
        self._layout_jump()
        self._snap()
        self.obs.append(Formation(self.t, "jump_ball", None, None, None, None))
        self.advance(3.0)
        winner: Side = self.rng.choice(SIDES)
        self.arrow = other_side(winner)
        self._ev(EventType.JUMP_BALL, winner, None, {"winner_team": winner})
        self._start_clock()
        self.ball = ("free", (0.0, 0.0))
        self.advance(0.7)
        idx = self._pick(winner)
        self._set_holder(winner, idx)
        self.off, self.mode, self.start_idx, self.fast = winner, "carry", idx, False
        self.sc_t0 = self.t

    def run(self) -> SimResult:
        self.obs.append(ClockState(0.0, 1, self.plen, False))
        self._layout_jump()
        self._snap()
        self.advance(2.0)
        self._ev(EventType.GAME_START)
        for side in SIDES:
            self._ev(EventType.STARTERS, side, None, {"players": list(self.lineup[side])})
        self.advance(4.0)
        p = 1
        while True:
            length = self.plen if p <= self.periods else self.otlen
            self._play_period(p, length)
            last = p >= self.periods and self.score["home"] != self.score["away"]
            if last or p >= self.periods + 8:
                break
            nxt_len = self.plen if p + 1 <= self.periods else self.otlen
            half = (p == self.periods // 2)
            self._layout_huddle()
            self.advance(3.0)
            self.period = p + 1
            self.clock = nxt_len
            self._emit_clock(nxt_len, False)
            self.advance((self.halftime_s if half else self.break_s) - 3.0)
            p += 1
        self._ev(EventType.GAME_END)
        self.advance(5.0)
        # Abschluss-Scoreboard (Score ist inzwischen angezeigt)
        self._sb_tick()
        truth = sorted(self.truth, key=lambda e: e.video_ts)
        for i, e in enumerate(truth, 1):
            e.seq = i
        obs = sorted(self.obs, key=lambda o: o.video_ts)
        return SimResult(
            truth=truth, observations=obs, scoreboard=self.scoreboard, final_score=dict(self.score),
            track_of=dict(self.track_of), track_stints={k: list(v) for k, v in self.track_stints.items()},
            points=dict(self.points), duration=self.t, seed=self.seed, periods_played=p,
        )


def simulate_game(seed: int, game: Game, rosters: dict[str, list[Player]], *, periods: int = 4,
                  period_length_s: float = 600.0, noise: Optional[SimNoise] = None,
                  frame_hz: float = 10.0, include_frames: bool = True, referees: int = 0,
                  break_s: float = 120.0, halftime_s: float = 600.0,
                  tech_foul_prob: float = 0.004) -> SimResult:
    """Simuliert ein Spiel. ``rosters`` ist nach "home"/"away" oder nach Team-ID geschluesselt.

    Bei Gleichstand nach ``periods`` folgen Verlaengerungen (``game.overtime_length_s``).
    Die ``truth`` haengt nur von ``seed``, ``game``, ``rosters``, ``periods``, ``period_length_s``
    (und den Pausen-/Foul-Parametern) ab, nie von ``noise`` oder den Frame-Optionen.
    """
    return _Sim(seed, game, rosters, periods, period_length_s, noise, frame_hz, include_frames,
                referees, break_s, halftime_s, tech_foul_prob).run()
