"""Benchmark: simuliertes Spiel -> GameSession -> Box-Score gegen Wahrheit.

Plan ("Genauigkeitsziele" automatisch, "Metriken"): Gemessen wird je Spieler und
Spiel der Anteil exakt richtiger Werte (``accuracy_report``), F1 je Ereignistyp
(Zeittoleranz) und die Latenz des Live-Spielstands.

Festlegungen
------------
* Verglichen werden die Spieler, die in der Wahrheit Einsatzzeit haben (sonst
  blaehen Bankspieler mit lauter Nullen den Anteil auf).
* Minuten: ``minutes`` = Anteil Spieler mit |Abweichung| <= 30 s (Ziel automatisch),
  ``minutes_10s`` = <= 10 s (Ziel nach Korrektur, nur zur Information).
  Das Ziel fuer ``minutes`` ist die Annahme ``MINUTES_TARGET_SHARE`` (Plan nennt nur die Toleranz).
* Nicht optisch beobachtbar sind Fouls, Schrittfehler, Offensivfouls und
  24-s-Verstoesse; sie kommen im Produkt von Konsole/Spielbericht. Option
  ``console_fouls=True`` (Default) speist die Wahrheits-FOULs als source CONSOLE
  ein. ``tov`` zaehlt alle Ballverluste, ``tov_obs`` nur optisch beobachtbare Arten
  (bad_pass, steal, out_of_bounds) auf beiden Seiten.
* Der Spielstand ("Spielstand 100 %") ist ``score`` (Endstand == Wahrheit).
* Latenz: ``board_to_log`` = von der ersten Tafelanzeige eines neuen Standes bis zum
  ersten SCOREBOARD-Event im Log (Ziel < 3 s, Plan: Latenzbudget); ``shot_to_log``
  zusaetzlich ab dem Wurf (enthaelt die Anzeigeverzoegerung der Tafel).
* Rollout (Plan): ``rollout_gate(alt, neu)`` ist nur wahr, wenn keine Kategorie
  schlechter wird (Anteil exakt richtig nicht kleiner; bei gleichem Anteil MAE nicht groesser).
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Optional

import numpy as np

from courtvision.domain.events import (STAT_EVENT_TYPES, Event, EventSource, EventStatus, EventType,
                                       TurnoverKind, effective_events)
from courtvision.live.session import GameSession
from courtvision.qa.review import accuracy_report, event_f1
from courtvision.sim import SimNoise, demo_setup, simulate_game
from courtvision.stats.boxscore import compute_boxscore
from courtvision.store.eventstore import EventStore

#: Ziele "automatisch" (Plan, Genauigkeitsziele). None: nur Information.
TARGETS: dict[str, Optional[float]] = {
    "score": 1.0, "pts": 0.95, "fga": 0.90, "fg3a": 0.90, "fta": None, "ftm": None,
    "reb": 0.80, "orb": None, "drb": None, "ast": 0.75, "stl": 0.70, "blk": 0.70,
    "tov": 0.70, "tov_obs": None, "minutes": 0.95, "minutes_10s": None,
}
MINUTES_TARGET_SHARE = 0.95
COUNT_CATS = ["pts", "fga", "fg3a", "fta", "ftm", "orb", "drb", "reb", "ast", "stl", "blk", "tov"]
OBSERVABLE_TOV = {TurnoverKind.BAD_PASS.value, TurnoverKind.STEAL.value,
                  TurnoverKind.OUT_OF_BOUNDS.value}
_EPS = 1e-9


def _tov_kind(e: Event) -> str:
    k = e.payload.get("kind")
    return getattr(k, "value", k)


def _obs_only(evs: Iterable[Event]) -> list[Event]:
    return [e for e in evs if e.type != EventType.TURNOVER or _tov_kind(e) in OBSERVABLE_TOV]


def _lines(box, pids, cats) -> dict[str, dict[str, float]]:
    return {p: {c: getattr(box.players[p], c) for c in cats} for p in pids}


def _latency(board: list[tuple[float, dict]], log: list[Event], truth: list[Event]) -> dict[str, list[float]]:
    """Latenzen je neuem Spielstand (siehe Modulkopf)."""
    sb = [(e.video_ts, e.payload) for e in log if e.type == EventType.SCOREBOARD]
    b2l: list[float] = []
    s2l: list[float] = []
    for side in ("home", "away"):
        seen = 0
        for ts, d in board:
            v = d.get(side)
            if v is None or v <= seen:
                continue
            seen = v
            hit = next((t for t, p in sb if p.get(side) is not None and p[side] >= v and t >= ts - 1e-9), None)
            if hit is not None:
                b2l.append(hit - ts)
        pts = 0
        scoring = sorted((e for e in truth if e.team == side and e.payload.get("made")
                          and e.type in (EventType.SHOT, EventType.FREE_THROW)), key=lambda e: e.video_ts)
        for e in scoring:
            pts += int(e.payload.get("value", 1)) if e.type == EventType.SHOT else 1
            hit = next((t for t, p in sb if p.get(side) is not None and p[side] >= pts), None)
            if hit is not None:
                s2l.append(hit - e.video_ts)
    return {"board_to_log": b2l, "shot_to_log": s2l}


def run_seed(seed: int, *, noise: Optional[SimNoise] = None, include_frames: bool = True,
             console_fouls: bool = True, periods: int = 4, period_length_s: float = 600.0,
             f1_tol_s: float = 1.5, minutes_tol_s: float = 30.0,
             session_kwargs: Optional[dict] = None) -> dict[str, Any]:
    """Ein simuliertes Spiel durch die komplette Live-Kette; Ergebnis je Kategorie."""
    demo = demo_setup()
    game = demo.game.model_copy(update={"periods": periods, "period_length_s": period_length_s})
    t0 = time.time()
    sim = simulate_game(seed, game, demo.rosters, periods=periods, period_length_s=period_length_s,
                        noise=noise, include_frames=include_frames)
    t_sim = time.time() - t0
    store = EventStore(":memory:")
    store.put_game(game)
    sess = GameSession(store, game, demo.rosters, **{"calibration": sim.zones, **(session_kwargs or {})})
    for side in ("home", "away"):
        st = next(e for e in sim.truth if e.type == EventType.STARTERS and e.team == side)
        sess.set_starters(side, st.payload["players"])
    t0 = time.time()
    sess.feed_many(sim.observations, sim.scoreboard)
    sess.finish()
    if console_fouls:  # Kampfgericht tippt Fouls (nicht optisch beobachtbar)
        for e in sim.truth:
            if e.type == EventType.FOUL:
                store.append(e.derive(supersedes=None, source=EventSource.CONSOLE,
                                      status=EventStatus.CONFIRMED, confidence=1.0))
    t_run = time.time() - t0
    auto = store.effective(game.id)

    bt = compute_boxscore(sim.truth, game, demo.rosters)
    ba = compute_boxscore(auto, game, demo.rosters)
    bt_o = compute_boxscore(_obs_only(sim.truth), game, demo.rosters)
    ba_o = compute_boxscore(_obs_only(auto), game, demo.rosters)
    played = [p for p, l in bt.players.items() if l.seconds_played > 0]

    rep = accuracy_report(_lines(ba, played, COUNT_CATS), _lines(bt, played, COUNT_CATS))["per_category"]
    cats: dict[str, dict[str, float]] = {
        c: {"share": v["exact_share"], "mae": v["mae"], "n": v["n"]} for c, v in rep.items()}
    rep_o = accuracy_report(_lines(ba_o, played, ["tov"]), _lines(bt_o, played, ["tov"]))["per_category"]["tov"]
    cats["tov_obs"] = {"share": rep_o["exact_share"], "mae": rep_o["mae"], "n": rep_o["n"]}
    diffs = [abs(ba.players[p].seconds_played - bt.players[p].seconds_played) for p in played]
    cats["minutes"] = {"share": float(np.mean([d <= minutes_tol_s + _EPS for d in diffs])),
                       "mae": float(np.mean(diffs)), "n": len(diffs)}
    cats["minutes_10s"] = {"share": float(np.mean([d <= 10.0 + _EPS for d in diffs])),
                           "mae": float(np.mean(diffs)), "n": len(diffs)}
    cats["score"] = {"share": float(ba.score == bt.score and ba.score == sim.final_score),
                     "mae": float(sum(abs(ba.score[s] - bt.score[s]) for s in ("home", "away"))), "n": 1}

    stat_t = lambda evs: [e for e in effective_events(evs) if e.type in STAT_EVENT_TYPES]  # noqa: E731
    f1 = event_f1(stat_t(auto), stat_t(sim.truth), tol_s=f1_tol_s)
    lat = _latency(sim.scoreboard, auto, sim.truth)
    return {"seed": seed, "categories": cats, "f1": f1, "latency": lat,
            "team_minutes_s": {"auto": sum(l.seconds_played for l in ba.players.values()),
                               "truth": sum(l.seconds_played for l in bt.players.values())},
            "score": {"auto": ba.score, "truth": bt.score}, "duration_s": sim.duration,
            "runtime_s": {"sim": t_sim, "session": t_run}, "n_obs": len(sim.observations)}


def _pool_categories(runs: list[dict], frames: bool = True) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c in runs[0]["categories"]:
        n = sum(r["categories"][c]["n"] for r in runs)
        share = sum(r["categories"][c]["share"] * r["categories"][c]["n"] for r in runs) / max(n, 1)
        mae = sum(r["categories"][c]["mae"] * r["categories"][c]["n"] for r in runs) / max(n, 1)
        tgt = TARGETS.get(c)
        if c in ("minutes", "minutes_10s") and not frames:
            tgt = None  # ohne FrameState keine Wechselerkennung: Minuten nicht bewertbar
        out[c] = {"share": share, "mae": mae, "n": n, "target": tgt,
                  "passed": None if tgt is None else bool(share + _EPS >= tgt)}
    return out


def _pool_f1(runs: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for t in sorted({t for r in runs for t in r["f1"]} - {"ALL"}):
        tp = sum(r["f1"].get(t, {}).get("tp", 0) for r in runs)
        fp = sum(r["f1"].get(t, {}).get("fp", 0) for r in runs)
        fn = sum(r["f1"].get(t, {}).get("fn", 0) for r in runs)
        p = tp / (tp + fp) if tp + fp else 1.0
        rc = tp / (tp + fn) if tp + fn else 1.0
        out[t] = {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
                  "tp": tp, "fp": fp, "fn": fn}
    return out


def _summ(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    a = np.asarray(xs, float)
    return {"n": len(xs), "mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def evaluate(sim_seeds: Iterable[int], noise: Optional[SimNoise] = None, include_frames: bool = True, *,
             console_fouls: bool = True, periods: int = 4, period_length_s: float = 600.0,
             f1_tol_s: float = 1.5, minutes_tol_s: float = 30.0,
             session_kwargs: Optional[dict] = None) -> dict[str, Any]:
    """Mehrere Seeds auswerten und zusammenfassen (Kategorien gepoolt ueber Spieler x Spiele).

    ``session_kwargs`` gehen an :class:`GameSession` (z. B. ``engine_config``, ``fusion_config``,
    ``plausibility``) -- so vergleicht man Konfigurationen und prueft ``rollout_gate``.

    Ergebnis: ``categories`` (share/mae/n/target/passed), ``f1`` (je Stat-Typ), ``latency``,
    ``all_passed`` (alle Ziele erfuellt), ``runs`` (Einzelergebnisse), ``meta``.
    """
    runs = [run_seed(s, noise=noise, include_frames=include_frames, console_fouls=console_fouls,
                     periods=periods, period_length_s=period_length_s, f1_tol_s=f1_tol_s,
                     minutes_tol_s=minutes_tol_s, session_kwargs=session_kwargs) for s in sim_seeds]
    if not runs:
        raise ValueError("keine Seeds")
    cats = _pool_categories(runs, include_frames)
    lat = {k: _summ([x for r in runs for x in r["latency"][k]]) for k in ("board_to_log", "shot_to_log")}
    lat["board_to_log"]["target_s"] = 3.0
    lat["board_to_log"]["passed"] = bool(lat["board_to_log"]["p95"] < 3.0) if lat["board_to_log"]["n"] else None
    return {
        "categories": cats, "f1": _pool_f1(runs), "latency": lat,
        "all_passed": all(v["passed"] for v in cats.values() if v["passed"] is not None),
        "runs": runs,
        "meta": {"seeds": [r["seed"] for r in runs], "include_frames": include_frames,
                 "noise": None if noise is None else vars(noise).copy() if hasattr(noise, "__dict__")
                 else str(noise), "console_fouls": console_fouls, "periods": periods,
                 "period_length_s": period_length_s},
    }


# ---------------------------------------------------------------------------
def _cats(report: dict) -> dict[str, dict[str, float]]:
    return report["categories"] if "categories" in report else report


def rollout_regressions(old_report: dict, new_report: dict) -> list[str]:
    """Kategorien, in denen die neue Version schlechter ist (oder nicht messbar)."""
    old, new = _cats(old_report), _cats(new_report)
    bad = []
    for c, o in old.items():
        n = new.get(c)
        if n is None:
            bad.append(f"{c}: fehlt im neuen Report")
        elif n["share"] < o["share"] - _EPS:
            bad.append(f"{c}: Anteil {o['share']:.3f} -> {n['share']:.3f}")
        elif abs(n["share"] - o["share"]) <= _EPS and n["mae"] > o["mae"] + _EPS:
            bad.append(f"{c}: MAE {o['mae']:.3f} -> {n['mae']:.3f}")
    return bad


def rollout_gate(old_report: dict, new_report: dict) -> bool:
    """Plan: ausgerollt wird nur, wenn keine Box-Score-Kategorie schlechter wird."""
    return not rollout_regressions(old_report, new_report)


def format_report(res: dict[str, Any]) -> str:
    m = res["meta"]
    lines = [f"Seeds {m['seeds']}  frames={m['include_frames']}  noise={m['noise']}  "
             f"console_fouls={m['console_fouls']}  periods={m['periods']}x{m['period_length_s']:.0f}s",
             f"{'Kategorie':<19}{'Anteil':>8}{'MAE':>8}{'n':>6}{'Ziel':>8}  Ergebnis"]
    for c, v in res["categories"].items():
        tgt = "-" if v["target"] is None else f"{v['target']:.0%}"
        verdict = "info" if v["passed"] is None else ("PASS" if v["passed"] else "FAIL")
        label = c + (" (+-30s)" if c == "minutes" else " (+-10s)" if c == "minutes_10s" else "")
        lines.append(f"{label:<19}{v['share']:>8.1%}{v['mae']:>8.2f}{v['n']:>6}{tgt:>8}  {verdict}")
    lines.append("Event-F1 (nur Stat-Typen):")
    for t, v in res["f1"].items():
        lines.append(f"  {t:<14} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} "
                     f"(tp {v['tp']}, fp {v['fp']}, fn {v['fn']})")
    for k, v in res["latency"].items():
        extra = "" if "target_s" not in v else f"  Ziel < {v['target_s']:.0f}s: {'PASS' if v['passed'] else 'FAIL'}"
        lines.append(f"Latenz {k}: n={v['n']} mean={v['mean']:.2f}s p95={v['p95']:.2f}s max={v['max']:.2f}s{extra}")
    lines.append("ALLE ZIELE ERFUELLT" if res["all_passed"] else "ZIELE VERFEHLT")
    return "\n".join(lines)
